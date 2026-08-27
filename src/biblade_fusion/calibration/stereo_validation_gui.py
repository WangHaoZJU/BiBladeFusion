"""PySide6 GUI for independent validation of a fixed D435i IR calibration."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biblade_fusion.calibration.stereo_assets import LatestStereoFrameMailbox
from biblade_fusion.calibration.stereo_gui import RawD435iInfraredCapture
from biblade_fusion.calibration.stereo_validation import (
    StereoValidationAssetSession,
    StereoValidationError,
    StereoValidationResult,
    StereoValidationThresholds,
    validate_stereo_asset_session,
)
from biblade_fusion.core.settings import RealSenseConfig, StereoRectificationConfig


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


def launch_stereo_validation_gui(
    target_path: str | Path,
    calibration_path: str | Path,
    output_dir: str | Path,
    realsense_config: RealSenseConfig,
    rectification_config: StereoRectificationConfig,
    thresholds: StereoValidationThresholds,
) -> int:
    """Launch capture and offline validation without exposing any calibration solve."""

    from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    class CaptureWorker(QObject):
        frame_available = Signal()
        failed = Signal(str)
        finished = Signal()

        def __init__(
            self,
            session: StereoValidationAssetSession,
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
        succeeded = Signal(object)
        failed = Signal(str)
        finished = Signal()

        def __init__(self, session: StereoValidationAssetSession) -> None:
            super().__init__()
            self.session = session

        @Slot()
        def run(self) -> None:
            try:
                self.succeeded.emit(validate_stereo_asset_session(self.session))
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                self.finished.emit()

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("BiBladeFusion · D435i IR 独立标定验证")
            self.resize(1500, 780)
            self.session: StereoValidationAssetSession | None = None
            self.mailbox: LatestStereoFrameMailbox | None = None
            self.capture_thread: QThread | None = None
            self.capture_worker: CaptureWorker | None = None
            self.analysis_thread: QThread | None = None
            self.analysis_worker: AnalysisWorker | None = None
            self.result: StereoValidationResult | None = None

            explanation = QLabel(
                "用途：采集未参与标定的新姿态，只验证当前固定内外参；程序不会重新标定。\n"
                f"至少保存 {thresholds.minimum_accepted_pairs} 组，建议 8–12 组，覆盖中心、边缘、"
                "不同距离和倾角。"
            )
            explanation.setStyleSheet("font-size:15px;padding:6px")
            self.left_label = QLabel("点击“1. 开始并连接相机”后显示左红外画面")
            self.right_label = QLabel("点击“1. 开始并连接相机”后显示右红外画面")
            for label in (self.left_label, self.right_label):
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(600, 400)
                label.setStyleSheet("background:#111;color:#ddd")
            images = QHBoxLayout()
            images.addWidget(self.left_label)
            images.addWidget(self.right_label)

            self.start_button = QPushButton("1. 开始并连接相机")
            self.start_button.clicked.connect(self.start_capture)
            self.capture_button = QPushButton("2. 保存当前同步双目图像")
            self.capture_button.setEnabled(False)
            self.capture_button.clicked.connect(self.accept_sample)
            self.analyze_button = QPushButton("3. 采集完成，离线验证固定标定参数")
            self.analyze_button.setEnabled(False)
            self.analyze_button.clicked.connect(self.start_analysis)
            controls = QHBoxLayout()
            controls.addWidget(self.start_button)
            controls.addWidget(self.capture_button)
            controls.addWidget(self.analyze_button)
            controls.addStretch()

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.addWidget(explanation)
            layout.addLayout(images)
            layout.addLayout(controls)
            self.setCentralWidget(central)
            self.setStatusBar(QStatusBar())
            self.statusBar().showMessage(
                "等待开始；尚未连接相机、创建验证资产或统计图像"
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
                Qt.TransformationMode.FastTransformation,
            )

        @Slot()
        def start_capture(self) -> None:
            if self.capture_thread is not None or self.analysis_thread is not None:
                return
            try:
                session = StereoValidationAssetSession.create(
                    output_dir,
                    target_path=target_path,
                    calibration_path=calibration_path,
                    image_size=(
                        realsense_config.infrared_width,
                        realsense_config.infrared_height,
                    ),
                    frames_per_second=realsense_config.frames_per_second,
                    serial_number=realsense_config.serial_number,
                    emitter_enabled=realsense_config.infrared_emitter_enabled,
                    rectification=rectification_config,
                    thresholds=thresholds,
                )
            except Exception as exc:
                QMessageBox.critical(self, "无法开始验证", str(exc))
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
            self.analyze_button.setEnabled(False)
            self.statusBar().showMessage(
                f"正在连接 D435i；保存计数从 0 开始；验证资产：{session.root}"
            )
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
                    f"实时预览；已保存 {self.session.raw_pair_count} 组验证图像；"
                    f"目录：{self.session.root}"
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
            if self.session is not None and self.session.raw_pair_count == 0:
                self.start_button.setEnabled(True)
                self.statusBar().showMessage("相机连接失败或未保存图像；可重新开始新会话")

        @Slot()
        def accept_sample(self) -> None:
            if self.mailbox is None or self.session is None:
                return
            frame = self.mailbox.snapshot()
            if frame is None:
                return
            try:
                pair_id = self.session.record_pair(frame)
            except StereoValidationError as exc:
                self.statusBar().showMessage(str(exc))
                return
            enough = self.session.raw_pair_count >= thresholds.minimum_accepted_pairs
            self.analyze_button.setEnabled(enough)
            readiness = (
                "可以开始离线验证"
                if enough
                else f"至少需要 {thresholds.minimum_accepted_pairs} 组"
            )
            self.statusBar().showMessage(
                f"已保存 {self.session.raw_pair_count} 组独立验证图像（{pair_id}）；"
                f"{readiness}"
            )

        @Slot()
        def start_analysis(self) -> None:
            if self.analysis_thread is not None or self.session is None:
                return
            self.capture_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            if self.capture_worker is not None:
                self.capture_worker.request_stop()
            self.statusBar().showMessage(
                f"正在离线检测、校正并验证 {self.session.raw_pair_count} 组图像；"
                "固定标定参数不会被修改……"
            )
            thread = QThread()
            worker = AnalysisWorker(self.session)
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

        @Slot(object)
        def analysis_succeeded(self, result: StereoValidationResult) -> None:
            self.result = result
            metrics = result.metrics
            title = "验证通过" if metrics.passed else "验证未通过"
            QMessageBox.information(
                self,
                title,
                f"结论：{'PASS' if metrics.passed else 'FAIL'}\n"
                f"接受/拒绝图像：{metrics.accepted_pair_count}/{metrics.rejected_pair_count}\n"
                f"公共角点：{metrics.common_corner_count}\n"
                f"校正后垂直视差 RMSE/P95/最大值："
                f"{metrics.vertical_disparity_rmse_px:.4f}/"
                f"{metrics.vertical_disparity_p95_px:.4f}/"
                f"{metrics.vertical_disparity_max_px:.4f} px\n"
                f"左/右重投影 RMSE：{metrics.left_reprojection_rmse_px:.4f}/"
                f"{metrics.right_reprojection_rmse_px:.4f} px\n"
                f"双目传递 RMSE：{metrics.stereo_transfer_rmse_px:.4f} px\n"
                "标定参数重新拟合：否\n"
                f"报告：{result.report_json}\n"
                f"极线检查图：{result.analysis_root / 'pairs'}",
            )

        @Slot(str)
        def analysis_failed(self, message: str) -> None:
            QMessageBox.critical(
                self,
                "独立验证失败",
                f"{message}\n\n固定标定文件和原始验证图像均未被修改。",
            )

        @Slot()
        def analysis_finished(self) -> None:
            if self.analysis_thread is not None:
                self.analysis_thread.deleteLater()
            self.analysis_thread = None
            self.analysis_worker = None
            count = self.session.raw_pair_count if self.session is not None else 0
            if self.result is None:
                self.analyze_button.setEnabled(count >= thresholds.minimum_accepted_pairs)
                self.statusBar().showMessage(
                    f"验证未完成；已保存 {count} 组原始资产，可再次执行离线验证"
                )
            else:
                conclusion = "PASS" if self.result.metrics.passed else "FAIL"
                self.statusBar().showMessage(
                    f"验证完成：{conclusion}；报告：{self.result.report_json}"
                )

        def closeEvent(self, event: Any) -> None:
            if self.analysis_thread is not None:
                QMessageBox.information(self, "正在验证", "请等待离线验证结束后再关闭窗口")
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
