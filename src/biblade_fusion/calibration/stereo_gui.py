"""PySide6 live D435i infrared stereo-calibration application."""

from __future__ import annotations

import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biblade_fusion.calibration.stereo_assets import (
    LatestStereoFrameMailbox,
    RawInfraredStereoFrame,
    StereoCalibrationAssetError,
    StereoCalibrationAssetSession,
    solve_stereo_asset_session,
)
from biblade_fusion.calibration.stereo_charuco import (
    DistortionModel,
    SolvedStereoCalibration,
    StereoCharucoBoard,
)
from biblade_fusion.core.settings import RealSenseConfig


class RawD435iInfraredCapture:
    """Acquire synchronized Y8 pairs without querying factory calibration."""

    def __init__(self, config: RealSenseConfig) -> None:
        self.config = config
        self.pipeline: Any | None = None
        self.rs: Any | None = None
        self.device_info: dict[str, str] = {}

    def open(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        stream = rs.config()
        if self.config.serial_number:
            stream.enable_device(self.config.serial_number)
        for index in (1, 2):
            stream.enable_stream(
                rs.stream.infrared,
                index,
                self.config.infrared_width,
                self.config.infrared_height,
                rs.format.y8,
                self.config.frames_per_second,
            )
        profile = pipeline.start(stream)
        device = profile.get_device()
        for name in ("serial_number", "name", "firmware_version", "usb_type_descriptor"):
            option = getattr(rs.camera_info, name, None)
            if option is not None and device.supports(option):
                self.device_info[name] = str(device.get_info(option))
        option_namespace = getattr(rs, "option", None)
        emitter_option = getattr(option_namespace, "emitter_enabled", None)
        if emitter_option is not None:
            sensor = profile.get_device().first_depth_sensor()
            if sensor.supports(emitter_option):
                sensor.set_option(
                    emitter_option,
                    1.0 if self.config.infrared_emitter_enabled else 0.0,
                )
        for _ in range(self.config.warmup_frames):
            pipeline.wait_for_frames(self.config.timeout_ms)
        self.rs = rs
        self.pipeline = pipeline

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def capture(self) -> RawInfraredStereoFrame:
        if self.pipeline is None:
            raise RuntimeError("D435i raw infrared capture is not open")
        frames = self.pipeline.wait_for_frames(self.config.timeout_ms)
        left = frames.get_infrared_frame(1)
        right = frames.get_infrared_frame(2)
        if not left or not right:
            raise RuntimeError("D435i returned an incomplete infrared stereo pair")
        return RawInfraredStereoFrame(
            left=np.asanyarray(left.get_data()).copy(),
            right=np.asanyarray(right.get_data()).copy(),
            left_frame_number=int(left.get_frame_number()),
            right_frame_number=int(right.get_frame_number()),
            left_timestamp_ms=float(left.get_timestamp()),
            right_timestamp_ms=float(right.get_timestamp()),
            timestamp_domain=str(left.get_frame_timestamp_domain()),
            captured_at_utc=datetime.now(UTC).isoformat(),
        )


def _preview(image: np.ndarray, frame_number: int) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.putText(
        output,
        f"RAW Y8  frame={frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 255, 80),
        2,
    )
    return output


def launch_stereo_calibration_gui(
    target_path: str | Path,
    output_dir: str | Path,
    realsense_config: RealSenseConfig,
) -> int:
    """Launch the live GUI. PySide6 is imported only for this optional command."""

    from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
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

    StereoCharucoBoard.read(target_path)

    class CaptureWorker(QObject):
        frame_available = Signal()
        failed = Signal(str)
        finished = Signal()

        def __init__(
            self,
            session: StereoCalibrationAssetSession,
            mailbox: LatestStereoFrameMailbox,
        ) -> None:
            super().__init__()
            self.session = session
            self.mailbox = mailbox
            self.stop_requested = threading.Event()
            self.camera = RawD435iInfraredCapture(realsense_config)

        @Slot()
        def run(self) -> None:
            try:
                self.camera.open()
                self.session.record_device_info(self.camera.device_info)
                while not self.stop_requested.is_set():
                    frame = self.camera.capture()
                    if self.mailbox.publish(frame):
                        self.frame_available.emit()
            except Exception as exc:
                self.session.mark_capture_failed(str(exc))
                self.failed.emit(str(exc))
            finally:
                self.camera.close()
                self.finished.emit()

        def request_stop(self) -> None:
            self.stop_requested.set()

    class AnalysisWorker(QObject):
        succeeded = Signal(object, str)
        failed = Signal(str)
        finished = Signal()

        def __init__(
            self,
            session: StereoCalibrationAssetSession,
            minimum_samples: int,
            selected_model: str,
        ) -> None:
            super().__init__()
            self.session = session
            self.minimum_samples = minimum_samples
            self.selected_model = selected_model

        @Slot()
        def run(self) -> None:
            try:
                _detection_run, result, output = solve_stereo_asset_session(
                    self.session,
                    minimum_samples=self.minimum_samples,
                    distortion_model=self.selected_model,
                    runtime_calibration_path=(
                        realsense_config.stereo_calibration_path
                        or Path("data/calibrations/d435i_ir_active.yaml")
                    ),
                )
                self.succeeded.emit(result, str(output))
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                self.finished.emit()

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("BiBladeFusion · D435i IR Stereo ChArUco Calibration")
            self.resize(1500, 760)
            self.session: StereoCalibrationAssetSession | None = None
            self.mailbox: LatestStereoFrameMailbox | None = None
            self.capture_thread: QThread | None = None
            self.capture_worker: CaptureWorker | None = None
            self.result: SolvedStereoCalibration | None = None
            self.analysis_thread: QThread | None = None
            self.analysis_worker: AnalysisWorker | None = None
            self.left_label = QLabel("点击“开始”后连接左红外相机")
            self.right_label = QLabel("点击“开始”后连接右红外相机")
            for label in (self.left_label, self.right_label):
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(600, 400)
                label.setStyleSheet("background:#111;color:#ddd")
            images = QHBoxLayout()
            images.addWidget(self.left_label)
            images.addWidget(self.right_label)
            self.start_button = QPushButton("开始")
            self.start_button.clicked.connect(self.start_capture)
            self.capture_button = QPushButton("保存最新同步原始双目帧")
            self.capture_button.setEnabled(False)
            self.capture_button.clicked.connect(self.accept_sample)
            self.minimum = QSpinBox()
            self.minimum.setRange(10, 100)
            self.minimum.setValue(20)
            self.distortion_model = QComboBox()
            self.distortion_model.addItem("Brown五参数（推荐）", DistortionModel.BROWN5.value)
            self.distortion_model.addItem("径向二参数", DistortionModel.RADIAL2.value)
            self.distortion_model.addItem("Rational八参数", DistortionModel.RATIONAL8.value)
            self.distortion_model.addItem("自动比较（独立验证集）", "auto")
            self.solve_button = QPushButton("采集完成：离线检测 + 张正友初始化 + 双目联合BA")
            self.solve_button.setEnabled(False)
            self.solve_button.clicked.connect(self.start_offline_analysis)
            controls = QHBoxLayout()
            controls.addWidget(self.start_button)
            controls.addWidget(self.capture_button)
            controls.addWidget(QLabel("最少样本数"))
            controls.addWidget(self.minimum)
            controls.addWidget(QLabel("畸变模型"))
            controls.addWidget(self.distortion_model)
            controls.addWidget(self.solve_button)
            controls.addStretch()
            central = QWidget()
            layout = QVBoxLayout(central)
            layout.addLayout(images)
            layout.addLayout(controls)
            self.setCentralWidget(central)
            self.setStatusBar(QStatusBar())
            self.statusBar().showMessage("等待开始；尚未连接相机、创建会话或统计样本")

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
            try:
                session = StereoCalibrationAssetSession.create(
                    output_dir,
                    target_path=target_path,
                    image_size=(
                        realsense_config.infrared_width,
                        realsense_config.infrared_height,
                    ),
                    frames_per_second=realsense_config.frames_per_second,
                    serial_number=realsense_config.serial_number,
                    emitter_enabled=realsense_config.infrared_emitter_enabled,
                )
            except Exception as exc:
                QMessageBox.critical(self, "无法开始", str(exc))
                return
            mailbox = LatestStereoFrameMailbox()
            thread = QThread()
            worker = CaptureWorker(session, mailbox)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.frame_available.connect(self.on_frame_available)
            worker.failed.connect(self.capture_failed)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self.capture_thread_finished)
            self.session = session
            self.mailbox = mailbox
            self.capture_thread = thread
            self.capture_worker = worker
            self.start_button.setEnabled(False)
            self.capture_button.setEnabled(False)
            self.solve_button.setEnabled(False)
            self.statusBar().showMessage(f"正在连接 D435i；统计从 0 开始；资产会话：{session.root}")
            thread.start()

        @Slot()
        def on_frame_available(self) -> None:
            if self.mailbox is None or self.session is None:
                return
            frame = self.mailbox.take_for_preview()
            if frame is None:
                return
            self.left_label.setPixmap(
                self.pixmap(_preview(frame.left, frame.left_frame_number), self.left_label)
            )
            self.right_label.setPixmap(
                self.pixmap(_preview(frame.right, frame.right_frame_number), self.right_label)
            )
            if self.analysis_thread is None and self.result is None:
                self.capture_button.setEnabled(True)
                self.statusBar().showMessage(
                    f"采集中：已保存 {self.session.raw_pair_count} 组；"
                    f"资产会话：{self.session.root}"
                )

        @Slot(str)
        def capture_failed(self, message: str) -> None:
            self.capture_button.setEnabled(False)
            QMessageBox.critical(self, "D435i采集失败", message)

        @Slot()
        def capture_thread_finished(self) -> None:
            if self.capture_thread is not None:
                self.capture_thread.deleteLater()
            self.capture_thread = None
            self.capture_worker = None
            if (
                self.result is None
                and self.session is not None
                and self.session.raw_pair_count == 0
            ):
                self.start_button.setEnabled(True)
                self.statusBar().showMessage("采集未开始或连接失败；可再次点击“开始”创建新会话")

        @Slot()
        def accept_sample(self) -> None:
            if self.mailbox is None or self.session is None:
                return
            frame = self.mailbox.snapshot()
            if frame is None:
                return
            try:
                pair_id = self.session.record_pair(frame)
            except StereoCalibrationAssetError as exc:
                self.statusBar().showMessage(str(exc))
                return
            self.solve_button.setEnabled(self.session.raw_pair_count >= self.minimum.value())
            self.statusBar().showMessage(
                f"已保存 {self.session.raw_pair_count} 组原始资产（{pair_id}）；"
                f"目录：{self.session.root}"
            )

        @Slot()
        def start_offline_analysis(self) -> None:
            if self.analysis_thread is not None or self.session is None:
                return
            minimum = self.minimum.value()
            selected = str(self.distortion_model.currentData())
            if selected == "auto" and minimum < 20:
                QMessageBox.critical(self, "无法开始", "自动比较至少需要20组原始样本")
                return
            self.capture_button.setEnabled(False)
            self.solve_button.setEnabled(False)
            self.minimum.setEnabled(False)
            self.distortion_model.setEnabled(False)
            self.statusBar().showMessage(
                f"正在对 {self.session.raw_pair_count} 组原始资产进行离线检测与求解……"
            )
            thread = QThread()
            worker = AnalysisWorker(self.session, minimum, selected)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.succeeded.connect(self.analysis_succeeded)
            worker.failed.connect(self.analysis_failed)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self.analysis_finished)
            self.analysis_thread = thread
            self.analysis_worker = worker
            thread.start()

        @Slot(object, str)
        def analysis_succeeded(
            self,
            result: SolvedStereoCalibration,
            output: str,
        ) -> None:
            self.result = result
            if self.capture_worker is not None:
                self.capture_worker.request_stop()
            metrics = self.result.metrics
            comparison = ""
            if self.result.model_comparison:
                rows = ["\n独立验证集比较："]
                rows.extend(
                    f"{item.model.value}: reproj={item.validation_reprojection_rmse_px:.4f}px, "
                    f"epi={item.validation_epipolar_rmse_px:.4f}px"
                    for item in self.result.model_comparison
                )
                comparison = "\n".join(rows)
            QMessageBox.information(
                self,
                "标定完成",
                f"资产会话: {self.session.root if self.session is not None else 'unknown'}\n"
                f"配置文件: {output}\n"
                f"选定模型: {self.result.distortion_model.value}\n"
                f"左目 RMS: {metrics.left_monocular_rms_px:.4f} px\n"
                f"右目 RMS: {metrics.right_monocular_rms_px:.4f} px\n"
                f"联合 RMS: {metrics.joint_stereo_rms_px:.4f} px\n"
                f"极线 RMSE/P95: {metrics.epipolar_rmse_px:.4f}/{metrics.epipolar_p95_px:.4f} px\n"
                f"基线: {self.result.calibration.baseline_m:.6f} m"
                f"{comparison}",
            )

        @Slot(str)
        def analysis_failed(self, message: str) -> None:
            QMessageBox.critical(
                self,
                "离线检测或标定失败",
                f"{message}\n\n原始双目资产未被删除或覆盖，可继续采集后重新分析。",
            )

        @Slot()
        def analysis_finished(self) -> None:
            if self.analysis_thread is not None:
                self.analysis_thread.deleteLater()
            self.analysis_thread = None
            self.analysis_worker = None
            completed = self.result is not None
            has_frame = self.mailbox is not None and self.mailbox.snapshot() is not None
            pair_count = self.session.raw_pair_count if self.session is not None else 0
            self.capture_button.setEnabled(not completed and has_frame)
            self.solve_button.setEnabled(not completed and pair_count >= self.minimum.value())
            self.minimum.setEnabled(not completed)
            self.distortion_model.setEnabled(not completed)
            if not completed:
                self.statusBar().showMessage(
                    f"离线分析未完成；原始资产共 {pair_count} 组，可补充后重试"
                )

        def closeEvent(self, event: Any) -> None:
            if self.analysis_thread is not None:
                QMessageBox.information(self, "正在处理", "请等待离线检测与标定完成后再关闭窗口")
                event.ignore()
                return
            if self.capture_worker is not None:
                self.capture_worker.request_stop()
            if self.session is not None:
                self.session.mark_capture_closed()
            event.accept()

    application = QApplication.instance() or QApplication(sys.argv)
    window = Window()
    window.show()
    result = application.exec()
    if window.capture_worker is not None:
        window.capture_worker.request_stop()
    if window.capture_thread is not None:
        window.capture_thread.quit()
        window.capture_thread.wait(6000)
    return result
