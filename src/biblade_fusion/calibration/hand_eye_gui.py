"""PySide6 read-only ES68 + D435i left-IR eye-in-hand calibration application."""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from biblade_fusion.acquisition import SynchronizedAcquirer, SynchronizedFrameBundle
from biblade_fusion.acquisition.errors import AcquisitionRejectedError
from biblade_fusion.calibration.charuco import (
    CharucoDetection,
    CharucoDetectionError,
    CharucoTargetDetector,
)
from biblade_fusion.calibration.hand_eye_solver import (
    HandEyeSample,
    solve_hand_eye,
    write_hand_eye_calibration,
    write_hand_eye_samples,
)
from biblade_fusion.calibration.stereo_charuco import StereoCharucoBoard
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    CharucoTargetConfig,
    HandEyeConfig,
    KinematicsConfig,
    RealSenseConfig,
    RobotConfig,
)
from biblade_fusion.devices.depth_camera import RealSenseD435i
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
        for point in detection.image_points_px:
            cv2.circle(output, tuple(np.rint(point).astype(int)), 3, color, -1)
    cv2.putText(output, message, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(
        output,
        "D435i raw LEFT IR (infrared/1)",
        (20, output.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 230, 80),
        2,
    )
    return output


def launch_hand_eye_calibration_gui(
    target_path: str | Path,
    output_dir: str | Path,
    robot_config: RobotConfig,
    realsense_config: RealSenseConfig,
    acquisition_config: AcquisitionConfig,
    hand_eye_config: HandEyeConfig,
    kinematics_config: KinematicsConfig,
) -> int:
    """Launch synchronized manual-pose capture and Daniilidis+LM solving."""

    from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QCloseEvent, QImage, QPixmap
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
    model = Es68KinematicModel.from_resources(
        joint_zero_offsets_rad=kinematics_config.joint_zero_offsets_rad
    )
    flange_t_tcp = load_es68_flange_t_tcp()
    root = Path(output_dir)
    sample_root = root / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)

    camera_config = realsense_config.model_copy(update={"enable_native_depth": False})

    class CaptureWorker(QObject):
        bundle = Signal(object)
        rejected = Signal(str)
        failed = Signal(str)
        finished = Signal()

        def __init__(self) -> None:
            super().__init__()
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
                        captured = acquirer.capture("hand-eye-preview", sequence)
                    except AcquisitionRejectedError as exc:
                        self.rejected.emit(str(exc))
                        continue
                    self.bundle.emit(captured)
                    sequence += 1
            except Exception as exc:
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

    class Window(QMainWindow):
        def __init__(self, worker: CaptureWorker) -> None:
            super().__init__()
            self.worker = worker
            self.setWindowTitle("BiBladeFusion · ES68 / D435i Left-IR Hand-Eye")
            self.resize(1380, 820)
            self.samples: list[HandEyeSample] = []
            self.current_bundle: SynchronizedFrameBundle | None = None
            self.current_detection: CharucoDetection | None = None
            self.current_fk_errors: tuple[float, float] | None = None

            self.image = QLabel("等待 ES68 与 D435i 左红外流")
            self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image.setMinimumSize(980, 620)
            self.image.setStyleSheet("background:#111;color:#ddd")
            self.details = QLabel(
                "求解链：base_T_flange(FK) · flange_T_left_ir · left_ir_T_board\n\n"
                "RTSI base_T_tcp 仅作独立一致性校验。\n"
                "每个姿态停止后请保持相同的最终趋近方向；避免改变 J6。"
            )
            self.details.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.details.setMinimumWidth(330)
            self.details.setWordWrap(True)
            images = QHBoxLayout()
            images.addWidget(self.image, stretch=1)
            images.addWidget(self.details)

            self.capture_button = QPushButton("保存当前同步样本")
            self.capture_button.setEnabled(False)
            self.capture_button.clicked.connect(self.accept_sample)
            self.minimum = QSpinBox()
            self.minimum.setRange(10, 100)
            self.minimum.setValue(max(20, hand_eye_config.minimum_samples))
            self.solve_button = QPushButton("Daniilidis 初值 + LM/BA")
            self.solve_button.setEnabled(False)
            self.solve_button.clicked.connect(self.solve)
            controls = QHBoxLayout()
            controls.addWidget(self.capture_button)
            controls.addWidget(QLabel("最少样本"))
            controls.addWidget(self.minimum)
            controls.addWidget(self.solve_button)
            controls.addStretch()

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.addLayout(images)
            layout.addLayout(controls)
            self.setCentralWidget(central)
            self.setStatusBar(QStatusBar())
            self.statusBar().showMessage(
                "请固定 ChArUco 板，改变相机三轴姿态、距离及图像区域；程序不会控制机械臂运动"
            )

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
                Qt.TransformationMode.SmoothTransformation,
            )

        @Slot(object)
        def on_bundle(self, bundle: SynchronizedFrameBundle) -> None:
            self.current_bundle = bundle
            try:
                detection = CharucoTargetDetector(target, bundle.stereo.calibration.left).detect(
                    bundle.stereo.left_ir
                )
                detection_message = (
                    f"corners={len(detection.charuco_ids)}  "
                    f"PnP={detection.reprojection_rmse_px:.3f}px"
                )
            except CharucoDetectionError as exc:
                detection = None
                detection_message = str(exc)
            self.current_detection = detection

            state = bundle.selected_robot_state
            base_t_flange = model.base_t_flange(state.joint_positions_rad)
            predicted_tcp = base_t_flange.compose(flange_t_tcp)
            translation_error = float(
                np.linalg.norm(predicted_tcp.translation_m - state.base_t_tcp.translation_m)
            )
            rotation_error = _rotation_error_deg(predicted_tcp.rotation, state.base_t_tcp.rotation)
            self.current_fk_errors = (translation_error, rotation_error)
            fk_valid = (
                translation_error <= hand_eye_config.maximum_fk_tcp_translation_error_m
                and rotation_error <= hand_eye_config.maximum_fk_tcp_rotation_error_deg
            )
            self.capture_button.setEnabled(detection is not None and fk_valid)
            annotated = _annotate_left_ir(bundle.stereo.left_ir, detection, detection_message)
            self.image.setPixmap(self.pixmap(annotated, self.image))

            joints_deg = np.degrees(state.joint_positions_rad)
            j6_span = (
                max(
                    np.degrees(sample.joint_positions_rad[5])
                    for sample in self.samples
                    if sample.joint_positions_rad is not None
                )
                - min(
                    np.degrees(sample.joint_positions_rad[5])
                    for sample in self.samples
                    if sample.joint_positions_rad is not None
                )
                if len(self.samples) >= 2
                else 0.0
            )
            self.details.setText(
                "求解相机流：D435i infrared/1（原始左红外）\n"
                "机器人位姿：HoloRobot ES68 709姿态标定FK\n\n"
                f"已保存：{len(self.samples)}\n"
                f"当前帧：{bundle.stereo.frame_number}\n"
                f"同步窗口：{bundle.metrics.bracket_ms:.2f} ms\n"
                f"FK/TCP 平移差：{translation_error * 1000:.3f} mm\n"
                f"FK/TCP 旋转差：{rotation_error:.4f} deg\n"
                f"当前 J6：{joints_deg[5]:.3f} deg\n"
                f"已采 J6 跨度：{j6_span:.3f} deg\n\n"
                "只有 ChArUco PnP 和 FK/TCP 校验同时合格时才能保存。"
            )

        @Slot(str)
        def on_rejected(self, message: str) -> None:
            self.capture_button.setEnabled(False)
            self.statusBar().showMessage(f"同步样本暂不可用：{message}")

        @Slot()
        def accept_sample(self) -> None:
            if (
                self.current_bundle is None
                or self.current_detection is None
                or self.current_fk_errors is None
            ):
                return
            bundle = self.current_bundle
            detection = self.current_detection
            if any(sample.frame_number == bundle.stereo.frame_number for sample in self.samples):
                self.statusBar().showMessage("当前 D435i 帧已经保存，请改变 ES68 姿态")
                return
            state = bundle.selected_robot_state
            sample_id = f"left_ir_{bundle.stereo.frame_number:08d}"
            sample = HandEyeSample(
                sample_id=sample_id,
                base_t_flange=model.base_t_flange(state.joint_positions_rad),
                left_ir_t_target=detection.left_ir_t_target,
                source_session=str(root.resolve()),
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
                selected_robot_state_offset_ms=(bundle.metrics.selected_robot_state_offset_ms),
                controller_time_s=state.controller_time_s,
                robot_mode=state.robot_mode,
                safety_status=state.safety_status,
                fk_tcp_translation_error_m=self.current_fk_errors[0],
                fk_tcp_rotation_error_deg=self.current_fk_errors[1],
            )
            sample_dir = sample_root / f"{len(self.samples):03d}_{sample_id}"
            sample_dir.mkdir(parents=True)
            cv2.imwrite(str(sample_dir / "left_ir.png"), bundle.stereo.left_ir)
            cv2.imwrite(str(sample_dir / "right_ir_audit.png"), bundle.stereo.right_ir)
            write_hand_eye_samples(sample_dir / "sample.json", [sample])
            self.samples.append(sample)
            self.solve_button.setEnabled(len(self.samples) >= self.minimum.value())
            self.statusBar().showMessage(
                f"已保存 {len(self.samples)} 个同步样本；继续改变三轴姿态、距离和成像位置"
            )

        @Slot()
        def solve(self) -> None:
            if self.current_bundle is None:
                return
            self.worker.pause()
            try:
                solution = solve_hand_eye(
                    self.samples,
                    hand_eye_config.model_copy(update={"minimum_samples": self.minimum.value()}),
                    method="daniilidis",
                    intrinsics=self.current_bundle.stereo.calibration.left,
                    refine=True,
                )
                sample_path = write_hand_eye_samples(root / "hand_eye_samples.json", self.samples)
                calibration_path = write_hand_eye_calibration(
                    root / "es68_d435i_left_ir_hand_eye.yaml",
                    solution,
                    intrinsics=self.current_bundle.stereo.calibration.left,
                    stereo_calibration_path=realsense_config.stereo_calibration_path,
                    target_path=target_path,
                )
            except Exception as exc:
                QMessageBox.critical(self, "手眼标定失败", str(exc))
                self.worker.resume()
                return
            optimization = solution.bundle_adjustment
            QMessageBox.information(
                self,
                "手眼标定完成",
                f"样本：{sample_path}\n"
                f"结果：{calibration_path}\n"
                f"方法：{solution.method}\n"
                f"样本数：{solution.sample_count}\n"
                f"闭环平移 RMS：{solution.translation_rmse_m * 1000:.3f} mm\n"
                f"闭环旋转 RMS/Max：{solution.rotation_rmse_deg:.4f}/"
                f"{solution.rotation_max_deg:.4f} deg\n"
                f"BA 分量 RMSE：{optimization.initial_rmse_px:.4f} → "
                f"{optimization.final_rmse_px:.4f} px",
            )
            self.worker.resume()

        def closeEvent(self, event: QCloseEvent) -> None:
            self.worker.stop()
            event.accept()

    application = QApplication.instance() or QApplication(sys.argv)
    worker = CaptureWorker()
    window = Window(worker)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.bundle.connect(window.on_bundle)
    worker.rejected.connect(window.on_rejected)
    worker.failed.connect(
        lambda message: QMessageBox.critical(window, "ES68/D435i 采集失败", message)
    )
    worker.finished.connect(thread.quit)
    thread.start()
    window.show()
    result = application.exec()
    worker.stop()
    thread.quit()
    thread.wait(7000)
    if thread.isRunning():
        raise RuntimeError("hand-eye capture worker did not stop within seven seconds")
    return result
