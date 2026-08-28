"""Read-only PySide6 supervisory console driven only by immutable snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biblade_fusion.supervision.snapshot import (
    StoredSupervisorySnapshot,
    SupervisoryTimeline,
    discover_supervisory_snapshots,
    load_snapshot_array,
)


@dataclass(frozen=True, slots=True)
class _DisplayArrays:
    robot_links: np.ndarray | None
    robot_mesh_vertices: np.ndarray | None
    robot_mesh_triangles: np.ndarray | None
    planned_path: np.ndarray | None
    actual_path: np.ndarray | None
    occupied: np.ndarray | None
    inflated: np.ndarray | None
    free: np.ndarray | None
    frontier: np.ndarray | None
    unknown: np.ndarray | None
    current_cloud: np.ndarray | None
    fused_cloud: np.ndarray | None
    fused_colors: np.ndarray | None
    left_ir: np.ndarray | None
    right_ir: np.ndarray | None
    depth_m: np.ndarray | None
    confidence: np.ndarray | None
    robot_self_mask: np.ndarray | None


def _load_display_arrays(stored: StoredSupervisorySnapshot) -> _DisplayArrays:
    snapshot = stored.snapshot
    return _DisplayArrays(
        robot_links=load_snapshot_array(stored, snapshot.robot.link_origins_base_m),
        robot_mesh_vertices=load_snapshot_array(
            stored, snapshot.robot.collision_mesh_vertices_base_m
        ),
        robot_mesh_triangles=load_snapshot_array(
            stored, snapshot.robot.collision_mesh_triangles
        ),
        planned_path=load_snapshot_array(stored, snapshot.robot.planned_tcp_path_base_m),
        actual_path=load_snapshot_array(stored, snapshot.robot.actual_tcp_path_base_m),
        occupied=load_snapshot_array(stored, snapshot.occupancy.occupied_centres_m),
        inflated=load_snapshot_array(stored, snapshot.occupancy.inflated_centres_m),
        free=load_snapshot_array(stored, snapshot.occupancy.free_centres_m),
        frontier=load_snapshot_array(stored, snapshot.occupancy.frontier_centres_m),
        unknown=load_snapshot_array(stored, snapshot.occupancy.unknown_centres_m),
        current_cloud=load_snapshot_array(stored, snapshot.reconstruction.current_points_m),
        fused_cloud=load_snapshot_array(stored, snapshot.reconstruction.fused_points_m),
        fused_colors=load_snapshot_array(stored, snapshot.reconstruction.fused_colors_rgb),
        left_ir=load_snapshot_array(stored, snapshot.sensor.left_ir),
        right_ir=load_snapshot_array(stored, snapshot.sensor.right_ir),
        depth_m=load_snapshot_array(stored, snapshot.sensor.depth_m),
        confidence=load_snapshot_array(stored, snapshot.sensor.confidence),
        robot_self_mask=load_snapshot_array(stored, snapshot.sensor.robot_self_mask),
    )


def launch_supervisory_console(
    source: str | Path,
    *,
    replay_interval_ms: int = 800,
    follow: bool = False,
    follow_poll_interval_ms: int = 1_000,
) -> int:
    """Launch the snapshot-only console; this function cannot command hardware."""

    if replay_interval_ms < 100:
        raise ValueError("replay_interval_ms must be at least 100")
    if follow_poll_interval_ms < 500:
        raise ValueError("follow_poll_interval_ms must be at least 500")
    source_path = Path(source).resolve()
    if follow and (
        not source_path.is_dir() or (source_path / "snapshot.json").is_file()
    ):
        raise ValueError(
            "Follow mode requires a timeline root whose child directories contain snapshots"
        )

    # PySide6 remains an optional dependency, imported only for the GUI entry point.
    from PySide6.QtCore import QPointF, Qt, QTimer
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    timeline: SupervisoryTimeline = discover_supervisory_snapshots(source_path)
    loaded_cache: dict[str, _DisplayArrays] = {}

    def arrays_for(stored: StoredSupervisorySnapshot) -> _DisplayArrays:
        if stored.content_sha256 not in loaded_cache:
            loaded_cache[stored.content_sha256] = _load_display_arrays(stored)
        return loaded_cache[stored.content_sha256]

    class OrbitCanvas(QWidget):
        def __init__(self, title: str) -> None:
            super().__init__()
            self._title = title
            self._stored: StoredSupervisorySnapshot | None = None
            self._arrays: _DisplayArrays | None = None
            self._last_position = None
            self._yaw = -0.65
            self._pitch = 0.48
            self._zoom = 1.0
            self.setMinimumSize(460, 340)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def set_scene(
            self,
            stored: StoredSupervisorySnapshot,
            arrays: _DisplayArrays,
        ) -> None:
            self._stored = stored
            self._arrays = arrays
            self.update()

        def reset_view(self) -> None:
            self._yaw, self._pitch, self._zoom = -0.65, 0.48, 1.0
            self.update()

        def _rotation(self) -> np.ndarray:
            cy, sy = np.cos(self._yaw), np.sin(self._yaw)
            cp, sp = np.cos(self._pitch), np.sin(self._pitch)
            yaw = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
            pitch = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)))
            return yaw @ pitch

        def _scene_points(self) -> tuple[np.ndarray, ...]:
            return ()

        def _projector(self):
            candidates = tuple(
                item
                for item in self._scene_points()
                if item is not None and item.ndim == 2 and len(item) and item.shape[1] == 3
            )
            if candidates:
                points = np.concatenate(candidates, axis=0)
                if len(points) > 50_000:
                    indices = np.linspace(0, len(points) - 1, 50_000, dtype=np.int64)
                    points = points[indices]
            else:
                points = np.asarray(((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0)))
            centre = np.mean(points, axis=0)
            rotated = (points - centre) @ self._rotation()
            span = np.maximum(np.ptp(rotated[:, :2], axis=0), 1e-3)
            scale = min(
                max(self.width() - 70, 1) / float(span[0]),
                max(self.height() - 80, 1) / float(span[1]),
            ) * self._zoom

            def project(values: np.ndarray) -> np.ndarray:
                transformed = (values - centre) @ self._rotation()
                screen = np.empty((len(values), 3), dtype=np.float64)
                screen[:, 0] = self.width() / 2.0 + transformed[:, 0] * scale
                screen[:, 1] = self.height() / 2.0 - transformed[:, 1] * scale
                screen[:, 2] = transformed[:, 2]
                return screen

            return project

        @staticmethod
        def _sample(points: np.ndarray | None, maximum: int) -> np.ndarray | None:
            if points is None or not len(points):
                return None
            if len(points) <= maximum:
                return points
            indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
            return points[indices]

        def _draw_cloud(
            self,
            painter: QPainter,
            project,
            points: np.ndarray | None,
            color: QColor,
            *,
            maximum: int = 12_000,
            width: float = 2.0,
        ) -> None:
            selected = self._sample(points, maximum)
            if selected is None:
                return
            screen = project(selected)
            inside = (
                (screen[:, 0] >= 0)
                & (screen[:, 0] < self.width())
                & (screen[:, 1] >= 0)
                & (screen[:, 1] < self.height())
            )
            polygon = QPolygonF(
                [QPointF(float(x), float(y)) for x, y in screen[inside, :2]]
            )
            painter.setPen(QPen(color, width))
            painter.drawPoints(polygon)

        def _draw_polyline(
            self,
            painter: QPainter,
            project,
            points: np.ndarray | None,
            pen: QPen,
        ) -> None:
            if points is None or len(points) < 2:
                return
            screen = project(points)
            painter.setPen(pen)
            for first, second in zip(screen[:-1], screen[1:], strict=True):
                painter.drawLine(
                    QPointF(float(first[0]), float(first[1])),
                    QPointF(float(second[0]), float(second[1])),
                )

        def _begin_paint(self) -> tuple[QPainter, object]:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#0d1218"))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QColor("#e7edf5"))
            painter.drawText(14, 24, self._title)
            return painter, self._projector()

        def _finish_paint(self, painter: QPainter) -> None:
            painter.setPen(QColor("#8f9dab"))
            painter.drawText(
                14,
                self.height() - 12,
                "左键拖动旋转 · 滚轮缩放 · 仅显示，不产生控制命令",
            )
            painter.end()

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                self._last_position = event.position()

        def mouseMoveEvent(self, event) -> None:
            if self._last_position is None:
                return
            delta = event.position() - self._last_position
            self._last_position = event.position()
            self._yaw += delta.x() * 0.008
            self._pitch = float(np.clip(self._pitch + delta.y() * 0.008, -1.5, 1.5))
            self.update()

        def mouseReleaseEvent(self, _event) -> None:
            self._last_position = None

        def wheelEvent(self, event) -> None:
            self._zoom = float(
                np.clip(self._zoom * np.exp(event.angleDelta().y() / 1200.0), 0.15, 10.0)
            )
            self.update()

    class SafetySceneCanvas(OrbitCanvas):
        def __init__(self) -> None:
            super().__init__("机器人、轨迹与三态占用安全场景")

        def _scene_points(self) -> tuple[np.ndarray, ...]:
            if self._arrays is None:
                return ()
            occupancy = self._stored.snapshot.occupancy if self._stored is not None else None
            bounds = (
                np.asarray((occupancy.bounds_min_m, occupancy.bounds_max_m))
                if occupancy is not None
                else None
            )
            values = (
                self._arrays.robot_links,
                self._arrays.robot_mesh_vertices,
                self._arrays.planned_path,
                self._arrays.actual_path,
                self._arrays.occupied,
                self._arrays.inflated,
                self._arrays.frontier,
                self._arrays.unknown,
                bounds,
            )
            return tuple(item for item in values if item is not None)

        def paintEvent(self, _event) -> None:
            painter, project = self._begin_paint()
            if self._stored is None or self._arrays is None:
                painter.drawText(20, 52, "没有安全场景快照")
                self._finish_paint(painter)
                return

            arrays = self._arrays
            occupancy = self._stored.snapshot.occupancy
            lower = np.asarray(occupancy.bounds_min_m, dtype=np.float64)
            upper = np.asarray(occupancy.bounds_max_m, dtype=np.float64)
            corners = np.asarray(
                [
                    (x, y, z)
                    for x in (lower[0], upper[0])
                    for y in (lower[1], upper[1])
                    for z in (lower[2], upper[2])
                ]
            )
            screen_corners = project(corners)
            painter.setPen(QPen(QColor(96, 119, 139, 130), 1.0, Qt.PenStyle.DotLine))
            for first, second in (
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 3),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
            ):
                painter.drawLine(
                    QPointF(float(screen_corners[first, 0]), float(screen_corners[first, 1])),
                    QPointF(float(screen_corners[second, 0]), float(screen_corners[second, 1])),
                )
            self._draw_cloud(
                painter, project, arrays.unknown, QColor(119, 130, 144, 75), maximum=5_000
            )
            self._draw_cloud(
                painter, project, arrays.free, QColor(76, 110, 132, 55), maximum=6_000
            )
            self._draw_cloud(
                painter, project, arrays.inflated, QColor(255, 169, 69, 115), maximum=12_000
            )
            self._draw_cloud(
                painter, project, arrays.occupied, QColor(242, 80, 80, 210), maximum=16_000
            )
            self._draw_cloud(
                painter, project, arrays.frontier, QColor(190, 113, 255, 210), maximum=8_000
            )
            planned_pen = QPen(QColor("#4db3ff"), 2.0, Qt.PenStyle.DashLine)
            actual_pen = QPen(QColor("#56df92"), 2.5)
            self._draw_polyline(painter, project, arrays.planned_path, planned_pen)
            self._draw_polyline(painter, project, arrays.actual_path, actual_pen)

            if arrays.robot_mesh_vertices is not None and arrays.robot_mesh_triangles is not None:
                vertices = project(arrays.robot_mesh_vertices)
                triangles = arrays.robot_mesh_triangles.astype(np.int64, copy=False)
                valid = (
                    np.all(triangles >= 0, axis=1)
                    & np.all(triangles < len(vertices), axis=1)
                )
                triangles = triangles[valid]
                if len(triangles) > 2_500:
                    triangle_indices = np.linspace(
                        0, len(triangles) - 1, 2_500, dtype=np.int64
                    )
                    triangles = triangles[triangle_indices]
                depth_order = np.argsort(np.mean(vertices[triangles, 2], axis=1))
                painter.setPen(QPen(QColor(196, 211, 223, 105), 0.8))
                painter.setBrush(QColor(129, 153, 171, 55))
                for triangle in triangles[depth_order]:
                    painter.drawPolygon(
                        QPolygonF(
                            [
                                QPointF(float(vertices[index, 0]), float(vertices[index, 1]))
                                for index in triangle
                            ]
                        )
                    )
                painter.setBrush(Qt.BrushStyle.NoBrush)

            if arrays.robot_links is not None and len(arrays.robot_links):
                links = project(arrays.robot_links)
                painter.setPen(QPen(QColor("#dce7f1"), 6.0))
                for first, second in zip(links[:-1], links[1:], strict=True):
                    painter.drawLine(
                        QPointF(float(first[0]), float(first[1])),
                        QPointF(float(second[0]), float(second[1])),
                    )
                painter.setPen(QPen(QColor("#69c6ff"), 10.0))
                painter.drawPoints(
                    QPolygonF([QPointF(float(x), float(y)) for x, y in links[:, :2]])
                )

            camera_pose = self._stored.snapshot.robot.camera_pose
            if camera_pose is not None:
                matrix = np.asarray(camera_pose.matrix, dtype=np.float64)
                origin = matrix[:3, 3]
                extent = np.asarray(self._stored.snapshot.occupancy.bounds_max_m) - np.asarray(
                    self._stored.snapshot.occupancy.bounds_min_m
                )
                depth = max(float(np.min(extent)) * 0.16, 0.04)
                local = np.asarray(
                    (
                        (0.0, 0.0, 0.0),
                        (-0.45 * depth, -0.30 * depth, depth),
                        (0.45 * depth, -0.30 * depth, depth),
                        (0.45 * depth, 0.30 * depth, depth),
                        (-0.45 * depth, 0.30 * depth, depth),
                    )
                )
                frustum = local @ matrix[:3, :3].T + origin
                screen = project(frustum)
                painter.setPen(QPen(QColor("#f2d45c"), 1.5))
                for corner in range(1, 5):
                    painter.drawLine(
                        QPointF(float(screen[0, 0]), float(screen[0, 1])),
                        QPointF(float(screen[corner, 0]), float(screen[corner, 1])),
                    )
                for first, second in ((1, 2), (2, 3), (3, 4), (4, 1)):
                    painter.drawLine(
                        QPointF(float(screen[first, 0]), float(screen[first, 1])),
                        QPointF(float(screen[second, 0]), float(screen[second, 1])),
                    )

            snapshot = self._stored.snapshot
            legend_items = ["占用=红"]
            if arrays.free is not None:
                legend_items.append("自由=蓝灰")
            if arrays.inflated is not None:
                legend_items.append("膨胀=橙")
            if arrays.frontier is not None:
                legend_items.append("前沿=紫")
            if arrays.unknown is not None:
                legend_items.append("显式未知=灰")
            else:
                legend_items.append("未绘制工作空间=UNKNOWN/BLOCK")
            if arrays.planned_path is not None:
                legend_items.append("历史目标端点折线=蓝虚线")
            if arrays.actual_path is not None:
                legend_items.append("实际轨迹=绿")
            if camera_pose is not None:
                legend_items.append("黄色视锥=姿态示意（非内参尺寸）")
            painter.setPen(QColor("#cbd5df"))
            painter.drawText(14, 46, "  ".join(legend_items[:3]))
            if len(legend_items) > 3:
                painter.drawText(14, 65, "  ".join(legend_items[3:]))
            clearance = snapshot.plan.minimum_clearance_m
            clearance_text = (
                f"{clearance:.3f} m" if clearance is not None else "未知"
            )
            painter.drawText(
                14,
                86,
                f"地图 {snapshot.occupancy.version} · {snapshot.occupancy.state} · "
                f"最小间隙 {clearance_text}",
            )
            self._finish_paint(painter)

    class ReconstructionCanvas(OrbitCanvas):
        def __init__(self) -> None:
            super().__init__("未知叶片多视角融合与覆盖质量")

        def _scene_points(self) -> tuple[np.ndarray, ...]:
            if self._arrays is None:
                return ()
            values = (self._arrays.current_cloud, self._arrays.fused_cloud)
            return tuple(item for item in values if item is not None)

        def paintEvent(self, _event) -> None:
            painter, project = self._begin_paint()
            if self._stored is None or self._arrays is None:
                painter.drawText(20, 52, "没有重建快照")
                self._finish_paint(painter)
                return

            arrays = self._arrays
            fused = self._sample(arrays.fused_cloud, 22_000)
            colors = arrays.fused_colors
            if fused is not None and colors is not None and len(colors) == len(arrays.fused_cloud):
                if len(arrays.fused_cloud) > len(fused):
                    indices = np.linspace(
                        0, len(arrays.fused_cloud) - 1, len(fused), dtype=np.int64
                    )
                    colors = colors[indices]
                screen = project(fused)
                painter.setPen(QPen(QColor("#73d6e8"), 2.0))
                for point, color in zip(screen, colors, strict=True):
                    red, green, blue = np.clip(color, 0, 255).astype(np.uint8)
                    painter.setPen(QColor(int(red), int(green), int(blue), 210))
                    painter.drawPoint(QPointF(float(point[0]), float(point[1])))
            else:
                self._draw_cloud(
                    painter,
                    project,
                    fused,
                    QColor(90, 213, 230, 210),
                    maximum=22_000,
                )
            self._draw_cloud(
                painter,
                project,
                arrays.current_cloud,
                QColor(255, 197, 92, 190),
                maximum=10_000,
                width=2.5,
            )

            model = self._stored.snapshot.reconstruction
            provenance = {
                "CURRENT_RUN_VERIFIED": "当前运行来源已验证",
                "INDEPENDENT_REFERENCE": "独立参考资产（未绑定当前运行）",
                "UNAVAILABLE": "无重建来源",
            }[model.provenance_status]
            painter.setPen(QColor("#cbd5df"))
            painter.drawText(
                14,
                47,
                f"融合=青/资产颜色  当前帧=黄 · 视角 {model.registered_view_count} · "
                f"模型 {model.model_version} · {provenance}",
            )
            painter.drawText(
                14,
                68,
                f"正面 {model.front_coverage:.1%}  反面 {model.back_coverage:.1%}  "
                f"正面鳍片 {model.fin_front_coverage:.1%}  "
                f"反面鳍片 {model.fin_back_coverage:.1%}",
            )
            self._finish_paint(painter)

    def image_pixmap(array: np.ndarray | None, width: int, height: int) -> QPixmap | None:
        if array is None:
            return None
        image = np.asarray(array)
        image = np.squeeze(image)
        if image.ndim != 2:
            return None
        finite = np.isfinite(image)
        if not np.any(finite):
            normalized = np.zeros(image.shape, dtype=np.uint8)
        elif image.dtype == np.uint8:
            normalized = np.where(finite, image, 0).astype(np.uint8)
        else:
            values = image[finite].astype(np.float64)
            low, high = np.percentile(values, (2.0, 98.0))
            if high <= low:
                high = low + 1.0
            normalized = np.zeros(image.shape, dtype=np.uint8)
            normalized[finite] = np.clip(
                (image[finite].astype(np.float64) - low) * 255.0 / (high - low),
                0.0,
                255.0,
            ).astype(np.uint8)
        contiguous = np.ascontiguousarray(normalized)
        qimage = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        return QPixmap.fromImage(qimage).scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    class SensorImage(QLabel):
        def __init__(self, title: str) -> None:
            super().__init__(f"{title}\n无数据")
            self._title = title
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(210, 150)
            self.setStyleSheet("background: #0d1218; color: #8f9dab; border: 1px solid #2b3744;")

        def set_array(self, array: np.ndarray | None) -> None:
            pixmap = image_pixmap(array, max(self.width() - 8, 100), max(self.height() - 28, 100))
            if pixmap is None:
                self.setPixmap(QPixmap())
                self.setText(f"{self._title}\n无数据")
                return
            self.setText("")
            self.setPixmap(pixmap)
            self.setToolTip(self._title)

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._index = 0
            self._playing = False
            self.setWindowTitle(
                "BiBladeFusion 统一监督台（只读快照/回放"
                + ("/目录跟随，非实时避障" if follow else "")
                + "）"
            )

            central = QWidget()
            root_layout = QVBoxLayout(central)
            root_layout.setContentsMargins(8, 8, 8, 8)

            warning = QLabel(
                "只读监督与证据回放 · 本界面未连接机器人命令端口 · "
                "不能授权、开始、暂停或停止真机运动"
            )
            warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warning.setStyleSheet(
                "padding: 8px; color: #ffdf8c; background: #3b2d12; "
                "font-weight: 700; border: 1px solid #80611f;"
            )
            root_layout.addWidget(warning)

            status_frame = QFrame()
            status_frame.setStyleSheet("background: #17202a; border: 1px solid #2b3744;")
            status_layout = QHBoxLayout(status_frame)
            self.state_label = QLabel()
            self.robot_label = QLabel()
            self.map_label = QLabel()
            self.plan_label = QLabel()
            for label in (self.state_label, self.robot_label, self.map_label, self.plan_label):
                label.setWordWrap(True)
                label.setMinimumWidth(180)
                status_layout.addWidget(label, 1)
            root_layout.addWidget(status_frame)

            replay_row = QHBoxLayout()
            self.previous_button = QPushButton("上一快照")
            self.play_button = QPushButton("播放回放")
            self.next_button = QPushButton("下一快照")
            self.position_slider = QSlider(Qt.Orientation.Horizontal)
            self.position_slider.setRange(0, len(timeline.snapshots) - 1)
            self.timeline_label = QLabel()
            replay_row.addWidget(self.previous_button)
            replay_row.addWidget(self.play_button)
            replay_row.addWidget(self.next_button)
            replay_row.addWidget(self.position_slider, 1)
            replay_row.addWidget(self.timeline_label)
            root_layout.addLayout(replay_row)

            vertical_splitter = QSplitter(Qt.Orientation.Vertical)
            scene_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.safety_scene = SafetySceneCanvas()
            self.reconstruction_scene = ReconstructionCanvas()
            scene_splitter.addWidget(self.safety_scene)
            scene_splitter.addWidget(self.reconstruction_scene)
            scene_splitter.setStretchFactor(0, 11)
            scene_splitter.setStretchFactor(1, 9)
            vertical_splitter.addWidget(scene_splitter)

            tabs = QTabWidget()
            sensor_tab = QWidget()
            sensor_layout = QGridLayout(sensor_tab)
            self.sensor_images = {
                "left_ir": SensorImage("左红外校正图"),
                "right_ir": SensorImage("右红外校正图"),
                "depth_m": SensorImage("深度图 / m"),
                "confidence": SensorImage("FoundationStereo置信度"),
                "robot_self_mask": SensorImage("机器人自遮罩"),
            }
            for column, widget in enumerate(self.sensor_images.values()):
                sensor_layout.addWidget(widget, 0, column)
            self.sensor_metrics = QLabel()
            self.sensor_metrics.setWordWrap(True)
            sensor_layout.addWidget(self.sensor_metrics, 1, 0, 1, len(self.sensor_images))
            tabs.addTab(sensor_tab, "传感器与深度质量")

            self.asset_table = QTableWidget(0, 5)
            self.asset_table.setHorizontalHeaderLabels(("名称", "类型", "版本", "SHA-256", "路径"))
            self.asset_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            tabs.addTab(self.asset_table, "数字资产与来源")

            self.event_table = QTableWidget(0, 4)
            self.event_table.setHorizontalHeaderLabels(("时间", "级别", "类别", "事件"))
            self.event_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            tabs.addTab(self.event_table, "回放资产事件（非控制器流）")

            diagnostic_tab = QWidget()
            diagnostic_layout = QVBoxLayout(diagnostic_tab)
            self.diagnostic_label = QLabel()
            self.diagnostic_label.setWordWrap(True)
            self.diagnostic_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            diagnostic_layout.addWidget(self.diagnostic_label)
            diagnostic_layout.addStretch(1)
            tabs.addTab(diagnostic_tab, "计划、标定与安全证据")

            vertical_splitter.addWidget(tabs)
            vertical_splitter.setStretchFactor(0, 7)
            vertical_splitter.setStretchFactor(1, 3)
            root_layout.addWidget(vertical_splitter, 1)
            self.setCentralWidget(central)

            self.timer = QTimer(self)
            self.timer.setInterval(replay_interval_ms)
            self.timer.timeout.connect(self._advance_replay)
            self.follow_timer = QTimer(self)
            self.follow_timer.setInterval(follow_poll_interval_ms)
            self.follow_timer.timeout.connect(self._refresh_follow)
            self.previous_button.clicked.connect(lambda: self._show_index(self._index - 1))
            self.next_button.clicked.connect(lambda: self._show_index(self._index + 1))
            self.play_button.clicked.connect(self._toggle_replay)
            self.position_slider.valueChanged.connect(self._show_index)
            if len(timeline.snapshots) == 1:
                self.previous_button.setEnabled(False)
                self.play_button.setEnabled(False)
                self.next_button.setEnabled(False)
                self.position_slider.setEnabled(False)
            self._show_index(0)
            if follow:
                self.follow_timer.start()

        def _refresh_follow(self) -> None:
            nonlocal timeline
            previous_hashes = tuple(
                item.content_sha256 for item in timeline.snapshots
            )
            try:
                discovered = discover_supervisory_snapshots(source_path)
            except ValueError as exc:
                self.timeline_label.setToolTip(
                    f"目录跟随保留上一完整时间线；本轮发现失败：{exc}"
                )
                return
            discovered_hashes = tuple(
                item.content_sha256 for item in discovered.snapshots
            )
            if discovered_hashes == previous_hashes:
                return
            if discovered_hashes[: len(previous_hashes)] != previous_hashes:
                self.timeline_label.setToolTip(
                    "目录跟随拒绝已发布快照被修改、删除或重新排序"
                )
                return
            was_at_end = self._index == len(timeline.snapshots) - 1
            timeline = discovered
            self.position_slider.setMaximum(len(timeline.snapshots) - 1)
            enabled = len(timeline.snapshots) > 1
            self.previous_button.setEnabled(enabled)
            self.play_button.setEnabled(enabled)
            self.next_button.setEnabled(enabled)
            self.position_slider.setEnabled(enabled)
            self.timeline_label.setToolTip(
                "只读发现原子发布的回放快照；不构成在线感知或实时避障"
            )
            self._show_index(len(timeline.snapshots) - 1 if was_at_end else self._index)

        def _toggle_replay(self) -> None:
            self._playing = not self._playing
            if self._playing:
                if self._index == len(timeline.snapshots) - 1:
                    self._show_index(0)
                self.timer.start()
                self.play_button.setText("暂停回放")
            else:
                self.timer.stop()
                self.play_button.setText("播放回放")

        def _advance_replay(self) -> None:
            if self._index >= len(timeline.snapshots) - 1:
                self._toggle_replay()
                return
            self._show_index(self._index + 1)

        def _show_index(self, index: int) -> None:
            index = int(np.clip(index, 0, len(timeline.snapshots) - 1))
            self._index = index
            if self.position_slider.value() != index:
                self.position_slider.blockSignals(True)
                self.position_slider.setValue(index)
                self.position_slider.blockSignals(False)
            stored = timeline.snapshots[index]
            arrays = arrays_for(stored)
            snapshot = stored.snapshot
            self.safety_scene.set_scene(stored, arrays)
            self.reconstruction_scene.set_scene(stored, arrays)

            state_colors = {
                "BLOCKED": "#ffbd59",
                "READY_FOR_EXTERNAL_APPROVAL": "#69c6ff",
                "EXECUTING": "#56df92",
                "STOPPED": "#f0d36a",
                "FAULT": "#ff6b70",
            }
            color = state_colors[snapshot.safety.system_state]
            self.state_label.setText(
                f"<b style='color:{color}'>系统 {snapshot.safety.system_state}</b><br>"
                f"查看器 {snapshot.safety.viewer_mode} / command capable: false"
            )
            self.robot_label.setText(
                f"<b>ES68状态</b><br>{snapshot.robot.robot_mode} · "
                f"安全 {snapshot.robot.safety_status}<br>"
                f"模型 {snapshot.robot.model_id} · "
                + (
                    "哈希匹配碰撞网格"
                    if arrays.robot_mesh_vertices is not None
                    else "UNVERIFIED关节骨架示意"
                )
            )
            self.map_label.setText(
                f"<b>占用图 {snapshot.occupancy.state}</b><br>"
                f"{snapshot.occupancy.version} · age {snapshot.occupancy.age_s:.2f}s<br>"
                f"{snapshot.occupancy.integrated_frame_count}帧 · "
                f"voxel {snapshot.occupancy.voxel_size_m:.3f}m"
            )
            plan_position = (
                "-"
                if snapshot.plan.current_view_index is None
                else f"{snapshot.plan.current_view_index + 1}/{snapshot.plan.total_view_count}"
            )
            self.plan_label.setText(
                f"<b>计划 {snapshot.plan.state}</b><br>{snapshot.plan.plan_id}<br>"
                f"进度 {plan_position} · 下一视点 {snapshot.plan.next_view_id or '-'} · "
                + (
                    "历史目标端点折线"
                    if arrays.planned_path is not None
                    else "无已验证轨迹资产"
                )
            )
            self.timeline_label.setText(
                f"{index + 1}/{len(timeline.snapshots)} · seq {snapshot.sequence} · "
                f"快照 {stored.content_sha256[:12]}"
                + (" · 跟随回放目录（非实时避障）" if follow else "")
            )

            sensor_arrays = {
                "left_ir": arrays.left_ir,
                "right_ir": arrays.right_ir,
                "depth_m": arrays.depth_m,
                "confidence": arrays.confidence,
                "robot_self_mask": arrays.robot_self_mask,
            }
            for name, widget in self.sensor_images.items():
                widget.set_array(sensor_arrays[name])
            latency = snapshot.sensor.inference_latency_ms
            sensor_frame = (
                str(snapshot.sensor.frame_number)
                if snapshot.sensor.frame_number is not None
                else "-"
            )
            dropped_frames = (
                str(snapshot.sensor.dropped_frame_count)
                if snapshot.sensor.dropped_frame_count is not None
                else "未记录"
            )
            quality = snapshot.sensor
            quality_text = "占用质量证据：未记录"
            if quality.occupancy_quality_evidence_sha256 is not None:
                quality_text = (
                    "占用质量证据："
                    f"有效深度 {quality.valid_depth_fraction:.1%}，"
                    f"双目有效 {quality.stereo_valid_fraction:.1%}，"
                    f"置信度接受 {quality.confidence_accepted_fraction:.1%}，"
                    f"接受区均值 {quality.mean_accepted_confidence:.3f}，"
                    f"LR阈值 {quality.lr_consistency_threshold_px:.2f}px；"
                    f"FK/TCP残差 {quality.fk_tcp_translation_error_m * 1_000.0:.2f}mm / "
                    f"{quality.fk_tcp_rotation_error_deg:.3f}deg；"
                    f"自遮罩 投影{quality.projected_robot_pixel_count} / "
                    f"实测{quality.measured_valid_pixel_count} / "
                    f"深度匹配{quality.depth_matched_pixel_count} / "
                    f"屏蔽{quality.masked_valid_pixel_count} / "
                    f"保留{quality.retained_valid_pixel_count} px；"
                    f"证据 {quality.occupancy_quality_evidence_sha256[:12]}"
                )
            self.sensor_metrics.setText(
                f"深度源：{snapshot.sensor.source}　帧号：{sensor_frame}　"
                f"推理延迟：{f'{latency:.1f} ms' if latency is not None else '未知'}　"
                f"丢帧：{dropped_frames}\n{quality_text}\n"
                "该页仅显示快照中的观测，不会重新运行推理或采集设备。"
            )

            self.asset_table.setRowCount(len(snapshot.assets))
            for row, asset in enumerate(snapshot.assets):
                values = (
                    asset.logical_name,
                    asset.kind,
                    asset.version or "-",
                    asset.sha256 or "-",
                    asset.path,
                )
                for column, value in enumerate(values):
                    self.asset_table.setItem(row, column, QTableWidgetItem(value))
            self.asset_table.resizeColumnsToContents()

            self.event_table.setRowCount(len(snapshot.events))
            for row, event in enumerate(snapshot.events):
                values = (
                    event.timestamp_utc.isoformat(),
                    event.severity,
                    event.category,
                    event.message,
                )
                for column, value in enumerate(values):
                    self.event_table.setItem(row, column, QTableWidgetItem(value))
            self.event_table.resizeColumnsToContents()

            reasons = tuple(
                dict.fromkeys((*snapshot.safety.blocking_reasons, *snapshot.plan.blocking_reasons))
            )
            reason_text = "；".join(reasons) if reasons else "无上报阻断原因"
            calibrations = "，".join(snapshot.safety.calibration_ids) or "未提供"
            occupancy_hash = snapshot.occupancy.content_sha256 or "未提供"
            feedback_age = (
                f"{snapshot.safety.feedback_age_ms:.1f} ms"
                if snapshot.safety.feedback_age_ms is not None
                else "未知"
            )
            reconstruction_reasons = (
                "；".join(snapshot.reconstruction.provenance_reasons) or "无"
            )
            self.diagnostic_label.setText(
                "<b>不可绕过的显示事实</b><br>"
                f"未知体素策略：{snapshot.safety.unknown_occupancy_policy}；"
                f"过期地图策略：{snapshot.safety.stale_occupancy_policy}；"
                f"反馈年龄：{feedback_age}<br>"
                f"阻断原因：{reason_text}<br>"
                f"标定资产：{calibrations}<br>"
                f"重建来源：{snapshot.reconstruction.provenance_status}；"
                f"来源说明：{reconstruction_reasons}<br>"
                f"占用图内容SHA-256：{occupancy_hash}<br>"
                f"快照文件：{stored.path}<br>"
                "本界面没有审批、执行、ServoJ、急停或任意关节/TCP写入入口。"
            )

    application = QApplication.instance() or QApplication([])
    window = Window()
    window.resize(1560, 980)
    window.show()
    return int(application.exec())
