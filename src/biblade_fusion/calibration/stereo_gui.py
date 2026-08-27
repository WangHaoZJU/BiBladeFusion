"""PySide6 live D435i infrared stereo-calibration application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biblade_fusion.calibration.stereo_charuco import (
    CharucoImageDetection,
    DistortionModel,
    SolvedStereoCalibration,
    StereoCharucoBoard,
    StereoCharucoDetector,
    StereoCharucoSample,
    compare_and_solve_stereo_charuco,
    solve_stereo_charuco,
    write_stereo_calibration,
)
from biblade_fusion.core.settings import RealSenseConfig


class RawD435iInfraredCapture:
    """Acquire synchronized Y8 pairs without querying factory calibration."""

    def __init__(self, config: RealSenseConfig) -> None:
        self.config = config
        self.pipeline: Any | None = None
        self.rs: Any | None = None

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

    def capture(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.pipeline is None:
            raise RuntimeError("D435i raw infrared capture is not open")
        frames = self.pipeline.wait_for_frames(self.config.timeout_ms)
        left = frames.get_infrared_frame(1)
        right = frames.get_infrared_frame(2)
        if not left or not right:
            raise RuntimeError("D435i returned an incomplete infrared stereo pair")
        return (
            np.asanyarray(left.get_data()).copy(),
            np.asanyarray(right.get_data()).copy(),
            int(left.get_frame_number()),
        )


def _annotate(image: np.ndarray, detection: CharucoImageDetection | None) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if detection is None:
        cv2.putText(output, "NO BOARD", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 60, 60), 2)
        return output
    for point in detection.image_points_px:
        cv2.circle(output, tuple(np.rint(point).astype(int)), 3, (30, 255, 80), -1)
    cv2.putText(
        output,
        f"corners={detection.corner_count}",
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

    target = StereoCharucoBoard.read(target_path)
    detector = StereoCharucoDetector(target)
    root = Path(output_dir)
    image_root = root / "samples"
    image_root.mkdir(parents=True, exist_ok=True)

    class CaptureWorker(QObject):
        frame = Signal(object, object, int)
        failed = Signal(str)
        finished = Signal()

        def __init__(self) -> None:
            super().__init__()
            self.running = True
            self.camera = RawD435iInfraredCapture(realsense_config)

        @Slot()
        def run(self) -> None:
            try:
                self.camera.open()
                while self.running:
                    left, right, number = self.camera.capture()
                    self.frame.emit(left, right, number)
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                self.camera.close()
                self.finished.emit()

        @Slot()
        def stop(self) -> None:
            self.running = False

    class Window(QMainWindow):
        stop_capture = Signal()

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("BiBladeFusion · D435i IR Stereo ChArUco Calibration")
            self.resize(1500, 760)
            self.samples: list[StereoCharucoSample] = []
            self.current: tuple[np.ndarray, np.ndarray, int] | None = None
            self.current_detections: tuple[
                CharucoImageDetection | None, CharucoImageDetection | None
            ] = (None, None)
            self.result: SolvedStereoCalibration | None = None
            self.left_label = QLabel("Waiting for left IR")
            self.right_label = QLabel("Waiting for right IR")
            for label in (self.left_label, self.right_label):
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(600, 400)
                label.setStyleSheet("background:#111;color:#ddd")
            images = QHBoxLayout()
            images.addWidget(self.left_label)
            images.addWidget(self.right_label)
            self.capture_button = QPushButton("采集当前双目样本")
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
            self.solve_button = QPushButton("张正友初始化 + 双目联合BA")
            self.solve_button.setEnabled(False)
            self.solve_button.clicked.connect(self.solve)
            controls = QHBoxLayout()
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
            self.statusBar().showMessage(
                "请覆盖中心、四角、不同距离及明显俯仰/偏航/滚转；仅双方同时检测合格时采集"
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

        @Slot(object, object, int)
        def on_frame(self, left: np.ndarray, right: np.ndarray, number: int) -> None:
            left_detection = detector.detect(left)
            right_detection = detector.detect(right)
            self.current = (left, right, number)
            self.current_detections = (left_detection, right_detection)
            self.left_label.setPixmap(self.pixmap(_annotate(left, left_detection), self.left_label))
            self.right_label.setPixmap(
                self.pixmap(_annotate(right, right_detection), self.right_label)
            )
            valid = (
                left_detection is not None
                and right_detection is not None
                and min(left_detection.corner_count, right_detection.corner_count)
                >= target.minimum_corners_per_camera
            )
            self.capture_button.setEnabled(valid)

        @Slot()
        def accept_sample(self) -> None:
            if self.current is None:
                return
            left, right, number = self.current
            sample = detector.detect_pair(f"frame_{number:08d}", left, right)
            if sample is None:
                self.statusBar().showMessage("当前双目检测未达到共同采样要求")
                return
            if any(item.sample_id == sample.sample_id for item in self.samples):
                self.statusBar().showMessage("该帧已经采集，请改变标定板姿态")
                return
            sample_dir = image_root / f"{len(self.samples):03d}_{sample.sample_id}"
            sample_dir.mkdir(parents=True)
            cv2.imwrite(str(sample_dir / "left_ir.png"), left)
            cv2.imwrite(str(sample_dir / "right_ir.png"), right)
            self.samples.append(sample)
            self.solve_button.setEnabled(len(self.samples) >= self.minimum.value())
            self.statusBar().showMessage(
                f"已采集 {len(self.samples)} 组；请继续改变距离、图像位置和三轴姿态"
            )

        @Slot()
        def solve(self) -> None:
            try:
                image_size = (
                    realsense_config.infrared_width,
                    realsense_config.infrared_height,
                )
                selected = str(self.distortion_model.currentData())
                if selected == "auto":
                    if self.minimum.value() < 20:
                        raise ValueError("自动比较至少需要20组样本，以保留独立验证视图")
                    self.result = compare_and_solve_stereo_charuco(
                        self.samples,
                        image_size,
                        target,
                        minimum_samples=max(20, self.minimum.value()),
                    )
                else:
                    self.result = solve_stereo_charuco(
                        self.samples,
                        image_size,
                        target,
                        minimum_samples=self.minimum.value(),
                        distortion_model=DistortionModel(selected),
                    )
                output = write_stereo_calibration(
                    root / "d435i_ir_stereo_calibration.yaml",
                    self.result,
                    [sample.sample_id for sample in self.samples],
                )
            except Exception as exc:
                QMessageBox.critical(self, "标定失败", str(exc))
                return
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
                f"配置文件: {output}\n"
                f"选定模型: {self.result.distortion_model.value}\n"
                f"左目 RMS: {metrics.left_monocular_rms_px:.4f} px\n"
                f"右目 RMS: {metrics.right_monocular_rms_px:.4f} px\n"
                f"联合 RMS: {metrics.joint_stereo_rms_px:.4f} px\n"
                f"极线 RMSE/P95: {metrics.epipolar_rmse_px:.4f}/{metrics.epipolar_p95_px:.4f} px\n"
                f"基线: {self.result.calibration.baseline_m:.6f} m"
                f"{comparison}",
            )

        def closeEvent(self, event: Any) -> None:
            self.stop_capture.emit()
            event.accept()

    application = QApplication.instance() or QApplication(sys.argv)
    window = Window()
    thread = QThread()
    worker = CaptureWorker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    window.stop_capture.connect(worker.stop)
    worker.frame.connect(window.on_frame)
    worker.failed.connect(lambda message: QMessageBox.critical(window, "D435i采集失败", message))
    worker.finished.connect(thread.quit)
    thread.start()
    window.show()
    result = application.exec()
    worker.stop()
    thread.quit()
    thread.wait(6000)
    return result
