"""PySide6 ES68 + D435i left-IR eye-in-hand calibration and validation GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Event
from typing import Any, Literal

import cv2
import numpy as np

from biblade_fusion.acquisition import SynchronizedAcquirer, SynchronizedFrameBundle
from biblade_fusion.acquisition.errors import AcquisitionRejectedError
from biblade_fusion.calibration.charuco import (
    CharucoDetection,
    CharucoDetectionError,
    CharucoTargetDetector,
)
from biblade_fusion.calibration.hand_eye_assets import (
    HandEyeAssetSession,
    HandEyeValidationResult,
    LatestHandEyeBundleMailbox,
    evaluate_hand_eye_validation,
    load_fixed_hand_eye_solution,
)
from biblade_fusion.calibration.hand_eye_solver import (
    HandEyeSample,
    HandEyeSolution,
    hand_eye_observability,
    solve_hand_eye,
)
from biblade_fusion.calibration.stereo_charuco import (
    StereoCharucoBoard,
    load_stereo_calibration,
)
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    CharucoTargetConfig,
    HandEyeConfig,
    KinematicsConfig,
    RealSenseConfig,
    RobotConfig,
)
from biblade_fusion.devices.depth_camera import RealSenseD435i, list_realsense_devices
from biblade_fusion.devices.robot import EliteReadOnlyRobot
from biblade_fusion.devices.thermal_camera import NullThermalCamera
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp


def _target_config(board: StereoCharucoBoard, quality: HandEyeConfig) -> CharucoTargetConfig:
    return quality.target.model_copy(
        update={
            "squares_x": board.squares_x,
            "squares_y": board.squares_y,
            "square_length_m": board.square_length_m,
            "marker_length_m": board.marker_length_m,
            "dictionary": board.dictionary_name,
            "legacy_pattern": board.legacy_pattern,
            "minimum_corners": board.minimum_corners_per_camera,
            "detector_params": board.detector_params,
        }
    )


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _annotate_left_ir(
    image: np.ndarray,
    detection: CharucoDetection | None,
    message: str,
) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    color = (40, 230, 80) if detection is not None else (255, 70, 70)
    if detection is not None:
        for identifier, point in zip(
            detection.charuco_ids, detection.image_points_px, strict=True
        ):
            center = tuple(np.rint(point).astype(int))
            cv2.circle(output, center, 4, color, -1)
            if int(identifier) % 10 == 0:
                cv2.putText(
                    output,
                    str(int(identifier)),
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )
    cv2.putText(output, message, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    return output


def _raw_left_ir(image: np.ndarray, frame_number: int) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.putText(
        output,
        f"RAW LEFT IR  frame={frame_number}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (40, 230, 80),
        2,
    )
    return output


def _pose_novelty(
    base_t_flange: Any,
    samples: list[HandEyeSample],
    config: HandEyeConfig,
) -> tuple[bool, float | None, float | None]:
    nearest_translation: float | None = None
    nearest_rotation: float | None = None
    nearest_score: float | None = None
    for sample in samples:
        relative = sample.base_t_flange.inverse().compose(base_t_flange)
        translation = float(np.linalg.norm(relative.translation_m))
        rotation = _rotation_error_deg(np.eye(3), relative.rotation)
        score = (
            translation / config.minimum_novel_translation_m
            + rotation / config.minimum_novel_rotation_deg
        )
        if nearest_score is None or score < nearest_score:
            nearest_score = score
            nearest_translation = translation
            nearest_rotation = rotation
        if (
            translation < config.minimum_novel_translation_m
            and rotation < config.minimum_novel_rotation_deg
        ):
            return False, translation, rotation
    return True, nearest_translation, nearest_rotation


def launch_hand_eye_calibration_gui(
    target_path: str | Path,
    output_dir: str | Path,
    robot_config: RobotConfig,
    realsense_config: RealSenseConfig,
    acquisition_config: AcquisitionConfig,
    hand_eye_config: HandEyeConfig,
    kinematics_config: KinematicsConfig,
    validation_calibration_path: str | Path | None = None,
) -> int:
    """Launch idle-first manual capture, Park+BA solve, and held-out validation."""

    from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    if robot_config.model != "es68":
        raise ValueError("hand-eye GUI requires robot.model=es68")
    if realsense_config.stereo_calibration_path is None:
        raise ValueError(
            "realsense.stereo_calibration_path must point to the user-calibrated IR YAML"
        )
    board = StereoCharucoBoard.read(target_path)
    target = _target_config(board, hand_eye_config)
    fixed_stereo = load_stereo_calibration(realsense_config.stereo_calibration_path)
    detector = CharucoTargetDetector(target, fixed_stereo.left)
    model = Es68KinematicModel.from_resources(
        joint_zero_offsets_rad=kinematics_config.joint_zero_offsets_rad
    )
    flange_t_tcp = load_es68_flange_t_tcp()
    camera_config = realsense_config.model_copy(update={"enable_native_depth": False})
    runtime_path = hand_eye_config.calibration_path or Path(
        "data/calibrations/es68_left_ir_hand_eye_active.yaml"
    )
    supplemental_solution = (
        load_fixed_hand_eye_solution(validation_calibration_path, hand_eye_config)
        if validation_calibration_path is not None
        else None
    )

    class CaptureWorker(QObject):
        frame_available = Signal()
        connected = Signal(object)
        rejected = Signal(str)
        failed = Signal(str)
        finished = Signal()

        def __init__(
            self,
            session: HandEyeAssetSession,
            mailbox: LatestHandEyeBundleMailbox,
        ) -> None:
            super().__init__()
            self.session = session
            self.mailbox = mailbox
            self._stop = Event()
            self._pause = Event()

        @Slot()
        def run(self) -> None:
            robot = EliteReadOnlyRobot(robot_config)
            camera = RealSenseD435i(camera_config)
            thermal = NullThermalCamera()
            try:
                robot.connect()
                camera.open()
                devices = list_realsense_devices()
                selected_device = next(
                    (
                        item
                        for item in devices
                        if realsense_config.serial_number is None
                        or item.serial_number == realsense_config.serial_number
                    ),
                    None,
                )
                connection = {
                    "robot_ip": str(robot_config.robot_ip),
                    "robot_controller_version": robot.controller_version(),
                    "d435i_serial_number": (
                        selected_device.serial_number if selected_device is not None else None
                    ),
                    "d435i_name": selected_device.name if selected_device is not None else None,
                    "d435i_product_line": (
                        selected_device.product_line if selected_device is not None else None
                    ),
                    "stereo_calibration_path": str(
                        Path(realsense_config.stereo_calibration_path).resolve()  # type: ignore[arg-type]
                    ),
                }
                self.session.record_connection_info(connection)
                self.connected.emit(connection)
                acquirer = SynchronizedAcquirer(
                    robot,
                    camera,
                    thermal,
                    acquisition_config,
                    require_thermal=False,
                )
                sequence = 0
                while not self._stop.is_set():
                    if self._pause.is_set():
                        self._stop.wait(0.05)
                        continue
                    try:
                        captured = acquirer.capture("hand-eye", sequence)
                    except AcquisitionRejectedError as exc:
                        self.rejected.emit(str(exc))
                        continue
                    if self.mailbox.publish(captured):
                        self.frame_available.emit()
                    sequence += 1
            except Exception as exc:
                self.session.mark_failed(str(exc))
                self.failed.emit(str(exc))
            finally:
                camera.close()
                robot.disconnect()
                self.finished.emit()

        def stop(self) -> None:
            self._stop.set()

        def pause(self) -> None:
            self._pause.set()

        def resume(self) -> None:
            self._pause.clear()

    class SolveWorker(QObject):
        succeeded = Signal(object, str)
        failed = Signal(str)
        finished = Signal()

        def __init__(
            self,
            session: HandEyeAssetSession,
            samples: list[HandEyeSample],
            minimum_samples: int,
        ) -> None:
            super().__init__()
            self.session = session
            self.samples = list(samples)
            self.minimum_samples = minimum_samples

        @Slot()
        def run(self) -> None:
            try:
                solution = solve_hand_eye(
                    self.samples,
                    hand_eye_config.model_copy(
                        update={"minimum_samples": self.minimum_samples}
                    ),
                    method="park",
                    intrinsics=fixed_stereo.left,
                    refine=True,
                )
                candidate = self.session.record_candidate(
                    solution, self.samples, fixed_stereo.left
                )
                self.succeeded.emit(solution, str(candidate))
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                self.finished.emit()

    class ValidationWorker(QObject):
        succeeded = Signal(object, str, str)
        failed = Signal(str)
        finished = Signal()

        def __init__(
            self,
            session: HandEyeAssetSession,
            solution: HandEyeSolution,
            samples: list[HandEyeSample],
        ) -> None:
            super().__init__()
            self.session = session
            self.solution = solution
            self.samples = list(samples)

        @Slot()
        def run(self) -> None:
            try:
                result = evaluate_hand_eye_validation(
                    self.solution,
                    self.samples,
                    fixed_stereo.left,
                    hand_eye_config,
                )
                report, published, _ = self.session.finalize_validation(
                    result,
                    self.samples,
                    runtime_path,
                    hand_eye_config,
                )
                self.succeeded.emit(
                    result,
                    str(report),
                    "" if published is None else str(published),
                )
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                self.finished.emit()

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("BiBladeFusion · ES68 / D435i 左红外手眼标定")
            self.resize(1740, 900)
            self.mode: Literal["idle", "training", "solving", "validation", "complete"] = (
                "idle"
            )
            self.session: HandEyeAssetSession | None = None
            self.mailbox: LatestHandEyeBundleMailbox | None = None
            self.capture_worker: CaptureWorker | None = None
            self.capture_thread: QThread | None = None
            self.analysis_worker: SolveWorker | ValidationWorker | None = None
            self.analysis_thread: QThread | None = None
            self.training_samples: list[HandEyeSample] = []
            self.validation_samples: list[HandEyeSample] = []
            self.solution: HandEyeSolution | None = None
            self.current_bundle: SynchronizedFrameBundle | None = None
            self.current_detection: CharucoDetection | None = None
            self.current_fk_errors: tuple[float, float] | None = None
            self.current_annotated: np.ndarray | None = None
            self.current_valid = False
            self.current_reason = "尚未连接"

            instruction = QLabel(
                (
                    "补充独立验证：参数已冻结，不会重新拟合。"
                    if supplemental_solution is not None
                    else "固定ChArUco板；用示教器改变ES68姿态并完全停稳。"
                )
                + "按 C 保存，Backspace 撤销最后一组。程序只读机器人状态，不会控制运动。"
            )
            instruction.setStyleSheet("font-size:15px;padding:5px")
            self.raw_image = QLabel("点击“1. 开始并连接”后显示左红外原图")
            self.detection_image = QLabel("角点检测结果将在此显示")
            for label in (self.raw_image, self.detection_image):
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(620, 520)
                label.setStyleSheet("background:#111;color:#ddd")
            self.details = QLabel("状态：等待开始")
            self.details.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.details.setMinimumWidth(360)
            self.details.setWordWrap(True)
            images = QHBoxLayout()
            images.addWidget(self.raw_image, stretch=1)
            images.addWidget(self.detection_image, stretch=1)
            images.addWidget(self.details)

            self.start_button = QPushButton(
                "1. 开始补充独立验证"
                if supplemental_solution is not None
                else "1. 开始并连接 ES68 + D435i"
            )
            self.start_button.clicked.connect(self.start_capture)
            self.capture_button = QPushButton("2. 保存当前同步样本（C）")
            self.capture_button.setEnabled(False)
            self.capture_button.clicked.connect(self.accept_sample)
            self.undo_button = QPushButton("撤销最后一组（Backspace）")
            self.undo_button.setEnabled(False)
            self.undo_button.clicked.connect(self.undo_last_sample)
            self.minimum = QSpinBox()
            self.minimum.setRange(10, 100)
            self.minimum.setValue(max(20, hand_eye_config.minimum_samples))
            self.minimum.setEnabled(supplemental_solution is None)
            self.minimum_label = QLabel("训练最少样本")
            self.solve_button = QPushButton("3. 训练完成：Park 初值 + LM/BA")
            self.solve_button.setEnabled(False)
            self.solve_button.clicked.connect(self.start_solve)
            self.validate_button = QPushButton(
                "3. 完成补充独立验证并更新证据"
                if supplemental_solution is not None
                else "4. 完成独立验证并发布"
            )
            self.validate_button.setEnabled(False)
            self.validate_button.clicked.connect(self.start_validation)
            if supplemental_solution is not None:
                self.minimum_label.setVisible(False)
                self.minimum.setVisible(False)
                self.solve_button.setVisible(False)
            controls = QHBoxLayout()
            controls.addWidget(self.start_button)
            controls.addWidget(self.capture_button)
            controls.addWidget(self.undo_button)
            controls.addWidget(self.minimum_label)
            controls.addWidget(self.minimum)
            controls.addWidget(self.solve_button)
            controls.addWidget(self.validate_button)
            controls.addStretch()

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.addWidget(instruction)
            layout.addLayout(images)
            layout.addLayout(controls)
            self.setCentralWidget(central)
            self.setStatusBar(QStatusBar())
            self.statusBar().showMessage("等待操作员点击开始；尚未连接或创建资产会话")

            self.capture_shortcut = QShortcut(QKeySequence("C"), self)
            self.capture_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self.capture_shortcut.activated.connect(self.accept_sample)
            self.undo_shortcut = QShortcut(QKeySequence("Backspace"), self)
            self.undo_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self.undo_shortcut.activated.connect(self.undo_last_sample)

        @staticmethod
        def pixmap(rgb: np.ndarray, label: QLabel) -> QPixmap:
            height, width, channels = rgb.shape
            image = QImage(
                rgb.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
            return QPixmap.fromImage(image).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

        @Slot()
        def start_capture(self) -> None:
            if self.capture_thread is not None or self.analysis_thread is not None:
                return
            session: HandEyeAssetSession | None = None
            try:
                session = HandEyeAssetSession.create(
                    output_dir,
                    target_path=target_path,
                    stereo_calibration_path=realsense_config.stereo_calibration_path,  # type: ignore[arg-type]
                    robot_config=robot_config,
                    realsense_config=realsense_config,
                    hand_eye_config=hand_eye_config,
                    kinematics_config=kinematics_config,
                    session_mode=(
                        "supplemental_validation"
                        if supplemental_solution is not None
                        else "training_and_validation"
                    ),
                )
                if supplemental_solution is not None:
                    session.bind_fixed_candidate(
                        validation_calibration_path,  # type: ignore[arg-type]
                        supplemental_solution,
                    )
            except Exception as exc:
                if session is not None:
                    session.mark_failed(str(exc))
                QMessageBox.critical(self, "无法创建手眼标定会话", str(exc))
                return
            mailbox = LatestHandEyeBundleMailbox()
            worker = CaptureWorker(session, mailbox)
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.frame_available.connect(self.on_frame_available)
            worker.connected.connect(self.on_connected)
            worker.rejected.connect(self.on_rejected)
            worker.failed.connect(self.on_capture_failed)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self.capture_finished)
            self.session = session
            self.mailbox = mailbox
            self.capture_worker = worker
            self.capture_thread = thread
            if supplemental_solution is None:
                self.mode = "training"
            else:
                self.mode = "validation"
                self.solution = supplemental_solution
            self.start_button.setEnabled(False)
            self.minimum.setEnabled(supplemental_solution is None)
            self.statusBar().showMessage(f"正在连接设备；资产会话：{session.root}")
            thread.start()

        @Slot(object)
        def on_connected(self, information: dict[str, object]) -> None:
            self.statusBar().showMessage(
                f"已连接 ES68 {information.get('robot_ip')} 与 D435i "
                f"{information.get('d435i_serial_number')}；等待合格姿态"
            )

        @Slot()
        def on_frame_available(self) -> None:
            if self.mailbox is None:
                return
            bundle = self.mailbox.take_for_preview()
            if bundle is None:
                return
            self.current_bundle = bundle
            detection: CharucoDetection | None = None
            try:
                detection = detector.detect(bundle.stereo.left_ir)
                detection_message = (
                    f"corners={len(detection.charuco_ids)}  "
                    f"PnP={detection.reprojection_rmse_px:.3f}px  "
                    f"ambiguity={detection.pose_ambiguity_ratio or float('inf'):.2f}"
                )
            except CharucoDetectionError as exc:
                detection_message = str(exc)
            self.current_detection = detection

            state = bundle.selected_robot_state
            base_t_flange = model.base_t_flange(state.joint_positions_rad)
            predicted_tcp = base_t_flange.compose(flange_t_tcp)
            translation_error = float(
                np.linalg.norm(predicted_tcp.translation_m - state.base_t_tcp.translation_m)
            )
            rotation_error = _rotation_error_deg(
                predicted_tcp.rotation, state.base_t_tcp.rotation
            )
            self.current_fk_errors = (translation_error, rotation_error)
            fk_valid = (
                translation_error <= hand_eye_config.maximum_fk_tcp_translation_error_m
                and rotation_error <= hand_eye_config.maximum_fk_tcp_rotation_error_deg
            )
            all_samples = [*self.training_samples, *self.validation_samples]
            novel, nearest_translation, nearest_rotation = _pose_novelty(
                base_t_flange, all_samples, hand_eye_config
            )
            self.current_valid = detection is not None and fk_valid and novel
            reasons: list[str] = []
            if detection is None:
                reasons.append("ChArUco检测未通过")
            if not fk_valid:
                reasons.append("FK/TCP不一致")
            if not novel:
                reasons.append("姿态与已保存样本过近")
            self.current_reason = "；".join(reasons) if reasons else "可保存"
            self.capture_button.setEnabled(
                self.current_valid and self.mode in {"training", "validation"}
            )
            self.current_annotated = _annotate_left_ir(
                bundle.stereo.left_ir, detection, detection_message
            )
            self.raw_image.setPixmap(
                self.pixmap(
                    _raw_left_ir(bundle.stereo.left_ir, bundle.stereo.frame_number),
                    self.raw_image,
                )
            )
            self.detection_image.setPixmap(
                self.pixmap(self.current_annotated, self.detection_image)
            )
            self._update_details(
                bundle,
                detection,
                translation_error,
                rotation_error,
                nearest_translation,
                nearest_rotation,
            )

        def _update_details(
            self,
            bundle: SynchronizedFrameBundle,
            detection: CharucoDetection | None,
            fk_translation: float,
            fk_rotation: float,
            nearest_translation: float | None,
            nearest_rotation: float | None,
        ) -> None:
            if len(self.training_samples) >= 2:
                observable = hand_eye_observability(self.training_samples)
                observability_text = (
                    f"平移跨度：{observable.translation_span_m * 1000:.1f} mm\n"
                    f"旋转跨度：{observable.rotation_span_deg:.1f} deg\n"
                    f"旋转轴多样性：{observable.rotation_axis_diversity:.3f}"
                )
            else:
                observability_text = "至少保存2组后显示可观测性"
            board_text = "标定板：未检测"
            if detection is not None:
                normal = detection.left_ir_t_target.rotation[:, 2]
                tilt = float(
                    np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
                )
                centroid = np.mean(detection.image_points_px, axis=0)
                board_text = (
                    f"角点/PnP：{len(detection.charuco_ids)} / "
                    f"{detection.reprojection_rmse_px:.3f} px\n"
                    f"板距离：{np.linalg.norm(detection.left_ir_t_target.translation_m):.3f} m\n"
                    f"板倾角：{tilt:.1f} deg\n"
                    f"角点中心：({centroid[0]:.0f}, {centroid[1]:.0f}) px"
                )
            nearest = (
                "无历史样本"
                if nearest_translation is None or nearest_rotation is None
                else f"{nearest_translation * 1000:.1f} mm / {nearest_rotation:.1f} deg"
            )
            phase = "训练" if self.mode in {"training", "solving"} else "独立验证"
            training_text = (
                f"固定候选训练样本：{self.solution.sample_count}\n"
                if supplemental_solution is not None and self.solution is not None
                else f"训练样本：{len(self.training_samples)} / {self.minimum.value()}\n"
            )
            self.details.setText(
                f"阶段：{phase}\n"
                f"当前状态：{self.current_reason}\n\n"
                f"{training_text}"
                f"验证样本：{len(self.validation_samples)} / "
                f"{hand_eye_config.validation_minimum_samples}\n"
                f"当前帧：{bundle.stereo.frame_number}\n"
                f"同步窗口：{bundle.metrics.bracket_ms:.2f} ms\n"
                f"FK/TCP差：{fk_translation * 1000:.3f} mm / {fk_rotation:.4f} deg\n"
                f"最近姿态差：{nearest}\n\n"
                f"{board_text}\n\n"
                f"{observability_text}\n\n"
                "保持标定板固定；使用J1–J5形成多轴旋转，第一轮避免大幅改变J6。"
            )

        @Slot(str)
        def on_rejected(self, message: str) -> None:
            self.current_valid = False
            self.capture_button.setEnabled(False)
            self.statusBar().showMessage(f"同步采集暂不可用：{message}")

        @Slot(str)
        def on_capture_failed(self, message: str) -> None:
            self.capture_button.setEnabled(False)
            QMessageBox.critical(self, "ES68/D435i连接或采集失败", message)

        @Slot()
        def capture_finished(self) -> None:
            if self.capture_thread is not None:
                self.capture_thread.deleteLater()
            self.capture_thread = None
            self.capture_worker = None
            if (
                self.mode not in {"complete", "solving"}
                and not self.training_samples
                and not self.validation_samples
            ):
                self.mode = "idle"
                self.start_button.setEnabled(True)
            elif self.mode != "complete":
                self.statusBar().showMessage(
                    "采集连接已结束；当前会话资产已封存。请排除故障后重新启动程序，"
                    "不要把不同会话的样本混合。"
                )

        @Slot()
        def accept_sample(self) -> None:
            if self.mode not in {"training", "validation"}:
                self.statusBar().showMessage("当前阶段不能保存样本")
                return
            if (
                not self.current_valid
                or self.current_bundle is None
                or self.current_detection is None
                or self.current_fk_errors is None
                or self.current_annotated is None
                or self.session is None
            ):
                self.statusBar().showMessage(f"样本未保存：{self.current_reason}")
                return
            bundle = self.current_bundle
            detection = self.current_detection
            state = bundle.selected_robot_state
            phase: Literal["training", "validation"] = (
                "training" if self.mode == "training" else "validation"
            )
            prefix = "train" if phase == "training" else "validate"
            sample = HandEyeSample(
                sample_id=f"{prefix}_left_ir_{bundle.stereo.frame_number:08d}",
                base_t_flange=model.base_t_flange(state.joint_positions_rad),
                left_ir_t_target=detection.left_ir_t_target,
                source_session=str(self.session.root.resolve()),
                charuco_corner_count=len(detection.charuco_ids),
                reprojection_rmse_px=detection.reprojection_rmse_px,
                pose_ambiguity_ratio=detection.pose_ambiguity_ratio,
                joint_positions_rad=state.joint_positions_rad,
                base_t_tcp_observed=state.base_t_tcp,
                charuco_ids=detection.charuco_ids,
                image_points_px=detection.image_points_px,
                object_points_m=detection.object_points_m,
                frame_number=bundle.stereo.frame_number,
                bracket_ms=bundle.metrics.bracket_ms,
                selected_robot_state_offset_ms=bundle.metrics.selected_robot_state_offset_ms,
                controller_time_s=state.controller_time_s,
                robot_mode=state.robot_mode,
                safety_status=state.safety_status,
                fk_tcp_translation_error_m=self.current_fk_errors[0],
                fk_tcp_rotation_error_deg=self.current_fk_errors[1],
            )
            active = (
                self.training_samples if phase == "training" else self.validation_samples
            )
            if any(item.frame_number == sample.frame_number for item in active):
                self.statusBar().showMessage("当前相机帧已保存，请改变机械臂姿态")
                return
            try:
                sample_root = self.session.record_sample(
                    phase, sample, bundle, self.current_annotated
                )
            except Exception as exc:
                QMessageBox.critical(self, "样本保存失败", str(exc))
                return
            active.append(sample)
            self.current_valid = False
            self.capture_button.setEnabled(False)
            self.undo_button.setEnabled(True)
            self.solve_button.setEnabled(
                self.mode == "training" and len(self.training_samples) >= self.minimum.value()
            )
            self.validate_button.setEnabled(
                self.mode == "validation"
                and len(self.validation_samples) >= hand_eye_config.validation_minimum_samples
            )
            self.statusBar().showMessage(
                f"已保存{phase}样本 {len(active)}：{sample.sample_id}；目录：{sample_root}"
            )

        @Slot()
        def undo_last_sample(self) -> None:
            if self.session is None or self.mode not in {"training", "validation"}:
                return
            phase: Literal["training", "validation"] = self.mode
            active = (
                self.training_samples if phase == "training" else self.validation_samples
            )
            if not active:
                return
            try:
                sample_id = self.session.exclude_last_sample(phase)
            except Exception as exc:
                QMessageBox.critical(self, "无法撤销", str(exc))
                return
            active.pop()
            self.solve_button.setEnabled(
                self.mode == "training" and len(self.training_samples) >= self.minimum.value()
            )
            self.validate_button.setEnabled(
                self.mode == "validation"
                and len(self.validation_samples) >= hand_eye_config.validation_minimum_samples
            )
            self.undo_button.setEnabled(bool(active))
            self.statusBar().showMessage(
                f"已将最后一组 {sample_id} 标记为撤销；原始资产仍保留"
            )

        @Slot()
        def start_solve(self) -> None:
            if (
                self.mode != "training"
                or self.session is None
                or len(self.training_samples) < self.minimum.value()
                or self.analysis_thread is not None
            ):
                return
            self.mode = "solving"
            self.capture_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.solve_button.setEnabled(False)
            self.minimum.setEnabled(False)
            if self.capture_worker is not None:
                self.capture_worker.pause()
            thread = QThread()
            worker = SolveWorker(
                self.session, self.training_samples, self.minimum.value()
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.succeeded.connect(self.solve_succeeded)
            worker.failed.connect(self.solve_failed)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self.analysis_finished)
            self.analysis_thread = thread
            self.analysis_worker = worker
            self.statusBar().showMessage("正在执行 Park-Martin 初值与联合 LM/BA……")
            thread.start()

        @Slot(object, str)
        def solve_succeeded(self, solution: HandEyeSolution, candidate: str) -> None:
            self.solution = solution
            self.mode = "validation"
            optimization = solution.bundle_adjustment
            QMessageBox.information(
                self,
                "候选手眼参数求解完成",
                f"方法：{solution.method}\n"
                f"训练样本：{solution.sample_count}\n"
                f"闭环平移RMSE：{solution.translation_rmse_m * 1000:.3f} mm\n"
                f"闭环旋转RMSE/Max：{solution.rotation_rmse_deg:.4f}/"
                f"{solution.rotation_max_deg:.4f} deg\n"
                f"BA RMSE：{optimization.initial_rmse_px:.4f} → "
                f"{optimization.final_rmse_px:.4f} px\n"
                f"候选文件：{candidate}\n\n"
                f"现在请采集至少 {hand_eye_config.validation_minimum_samples} 个新姿态。"
                "验证阶段不会重新拟合参数，通过后才自动发布。",
            )

        @Slot(str)
        def solve_failed(self, message: str) -> None:
            self.mode = "training"
            self.minimum.setEnabled(True)
            self.undo_button.setEnabled(bool(self.training_samples))
            self.solve_button.setEnabled(len(self.training_samples) >= self.minimum.value())
            QMessageBox.critical(self, "Park+BA求解失败", message)

        @Slot()
        def start_validation(self) -> None:
            if (
                self.mode != "validation"
                or self.session is None
                or self.solution is None
                or len(self.validation_samples) < hand_eye_config.validation_minimum_samples
                or self.analysis_thread is not None
            ):
                return
            self.capture_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.validate_button.setEnabled(False)
            if self.capture_worker is not None:
                self.capture_worker.pause()
            thread = QThread()
            worker = ValidationWorker(
                self.session, self.solution, self.validation_samples
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.succeeded.connect(self.validation_succeeded)
            worker.failed.connect(self.validation_failed)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self.analysis_finished)
            self.analysis_thread = thread
            self.analysis_worker = worker
            self.statusBar().showMessage("正在执行固定参数独立验证；不会重新优化手眼参数……")
            thread.start()

        @Slot(object, str, str)
        def validation_succeeded(
            self,
            result: HandEyeValidationResult,
            report: str,
            published: str,
        ) -> None:
            metrics = result.metrics
            if metrics.passed:
                self.mode = "complete"
                if self.capture_worker is not None:
                    self.capture_worker.stop()
            else:
                self.mode = "validation"
            QMessageBox.information(
                self,
                "独立验证通过" if metrics.passed else "独立验证未通过",
                f"结论：{'PASS' if metrics.passed else 'FAIL'}\n"
                f"样本：{metrics.sample_count}\n"
                f"平移RMSE/P95/Max：{metrics.translation_rmse_m * 1000:.3f}/"
                f"{metrics.translation_p95_m * 1000:.3f}/"
                f"{metrics.translation_max_m * 1000:.3f} mm\n"
                f"旋转RMSE/P95/Max：{metrics.rotation_rmse_deg:.4f}/"
                f"{metrics.rotation_p95_deg:.4f}/{metrics.rotation_max_deg:.4f} deg\n"
                f"重投影RMSE/P95/Max：{metrics.reprojection_rmse_px:.4f}/"
                f"{metrics.reprojection_p95_px:.4f}/{metrics.reprojection_max_px:.4f} px\n"
                "参数重新拟合：否\n"
                f"报告：{report}\n"
                f"运行配置：{published or '未发布；可补充新验证姿态后重试'}",
            )
            if not metrics.passed:
                self.undo_button.setEnabled(bool(self.validation_samples))
                self.validate_button.setEnabled(True)

        @Slot(str)
        def validation_failed(self, message: str) -> None:
            self.mode = "validation"
            self.undo_button.setEnabled(bool(self.validation_samples))
            self.validate_button.setEnabled(
                len(self.validation_samples) >= hand_eye_config.validation_minimum_samples
            )
            QMessageBox.critical(self, "独立验证执行失败", message)

        @Slot()
        def analysis_finished(self) -> None:
            if self.analysis_thread is not None:
                self.analysis_thread.deleteLater()
            self.analysis_thread = None
            self.analysis_worker = None
            if self.capture_worker is not None and self.mode in {"training", "validation"}:
                self.capture_worker.resume()
            if self.mode == "validation":
                self.statusBar().showMessage(
                    f"独立验证阶段：已保存 {len(self.validation_samples)} / "
                    f"{hand_eye_config.validation_minimum_samples} 个新姿态"
                )
            elif self.mode == "complete":
                self.statusBar().showMessage(
                    f"手眼标定与独立验证完成；已自动发布：{runtime_path}"
                )

        def closeEvent(self, event: QCloseEvent) -> None:
            if self.analysis_thread is not None:
                QMessageBox.information(self, "正在计算", "请等待当前求解或验证完成")
                event.ignore()
                return
            if self.capture_worker is not None:
                self.capture_worker.stop()
            if self.session is not None:
                self.session.mark_closed()
            event.accept()

    application = QApplication.instance() or QApplication(sys.argv)
    window = Window()
    window.show()
    result = application.exec()
    if window.capture_worker is not None:
        window.capture_worker.stop()
    if window.capture_thread is not None:
        window.capture_thread.quit()
        window.capture_thread.wait(7000)
        if window.capture_thread.isRunning():
            raise RuntimeError("hand-eye capture worker did not stop within seven seconds")
    return result
