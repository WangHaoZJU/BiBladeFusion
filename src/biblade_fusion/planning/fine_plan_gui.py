"""PySide6 orbit viewer for immutable fine-plan inspection artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biblade_fusion.storage.coarse_model import read_coarse_model_summary
from biblade_fusion.storage.fine_plan_inspection import read_fine_plan_inspection
from biblade_fusion.workflows.fine_plan_inspection import REGION_COLORS


def _inspection_scene(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    stored = read_fine_plan_inspection(path)
    payload = stored.metadata
    coarse = read_coarse_model_summary(payload["source"]["coarse_model"])
    points_record = coarse.metadata["files"]["patch_points_m"]
    offsets_record = coarse.metadata["files"]["patch_offsets"]
    points = np.load(coarse.root / points_record["path"], allow_pickle=False)
    offsets = np.load(coarse.root / offsets_record["path"], allow_pickle=False)
    colors = np.empty((len(points), 3), dtype=np.uint8)
    labels = np.empty((len(points), 2), dtype=object)
    for index, patch in enumerate(coarse.metadata["surface"]["patches"]):
        start, end = int(offsets[index]), int(offsets[index + 1])
        colors[start:end] = REGION_COLORS.get(str(patch["region"]), (180, 180, 180))
        labels[start:end, 0] = str(patch["side"])
        labels[start:end, 1] = str(patch["region"])
    return payload, points, colors, labels


def launch_fine_plan_inspection_gui(path: str | Path) -> int:
    """Open a read-only interactive inspection window."""

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    payload, points, colors, labels = _inspection_scene(path)
    views = payload["views"]

    class OrbitCanvas(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setMinimumSize(760, 620)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.yaw = -0.65
            self.pitch = 0.52
            self.zoom = 1.0
            self.last_position = None
            self.side_filter = "all"
            self.region_filter = "all"
            self.show_frusta = True
            self.show_normals = True
            self.selected_view_id: str | None = None

        def reset_view(self) -> None:
            self.yaw, self.pitch, self.zoom = -0.65, 0.52, 1.0
            self.update()

        def _rotation(self) -> np.ndarray:
            cy, sy = np.cos(self.yaw), np.sin(self.yaw)
            cp, sp = np.cos(self.pitch), np.sin(self.pitch)
            yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
            return yaw @ pitch

        def _filtered_point_mask(self) -> np.ndarray:
            mask = np.ones(len(points), dtype=bool)
            if self.side_filter != "all":
                mask &= labels[:, 0] == self.side_filter
            if self.region_filter != "all":
                mask &= labels[:, 1] == self.region_filter
            return mask

        def _filtered_views(self) -> list[dict]:
            return [
                item
                for item in views
                if (self.side_filter == "all" or item["side"] == self.side_filter)
                and (self.region_filter == "all" or item["region"] == self.region_filter)
            ]

        def _projector(self, selected_points: np.ndarray):
            # Keep fitting anchored to the blade so distant cameras do not make the
            # thin-wall geometry unreadably small.
            centre = (
                np.mean(selected_points, axis=0)
                if len(selected_points)
                else np.mean(points, axis=0)
            )
            rotated = (selected_points - centre) @ self._rotation()
            span = np.ptp(rotated[:, :2], axis=0) if len(rotated) else np.ones(2)
            base_scale = min(
                max(self.width() - 50, 1) / max(float(span[0]), 1e-6),
                max(self.height() - 50, 1) / max(float(span[1]), 1e-6),
            )
            scale = base_scale * self.zoom

            def project(values: np.ndarray) -> np.ndarray:
                transformed = (values - centre) @ self._rotation()
                result = np.empty((len(values), 3), dtype=np.float64)
                result[:, 0] = self.width() / 2.0 + transformed[:, 0] * scale
                result[:, 1] = self.height() / 2.0 - transformed[:, 1] * scale
                result[:, 2] = transformed[:, 2]
                return result

            return project

        def paintEvent(self, _event) -> None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#11151b"))
            mask = self._filtered_point_mask()
            selected_points = points[mask]
            selected_colors = colors[mask]
            if not len(selected_points):
                painter.setPen(QColor("#e7edf4"))
                painter.drawText(24, 36, "当前筛选没有点")
                return
            project = self._projector(selected_points)
            projected = project(selected_points)
            sample = np.linspace(0, len(projected) - 1, min(len(projected), 14_000), dtype=np.int64)
            order = sample[np.argsort(projected[sample, 2])]
            for index in order:
                red, green, blue = selected_colors[index]
                painter.setPen(QColor(int(red), int(green), int(blue), 205))
                painter.drawPoint(int(projected[index, 0]), int(projected[index, 1]))

            for item in self._filtered_views():
                matrix = np.asarray(item["base_T_left_ir"], dtype=np.float64)
                camera = matrix[:3, 3]
                target = np.asarray(item["target_m"], dtype=np.float64)
                normal = np.asarray(item["outward_normal"], dtype=np.float64)
                width, height = item["footprint_m"]
                corners = np.asarray(
                    [
                        target - matrix[:3, 0] * width / 2 - matrix[:3, 1] * height / 2,
                        target + matrix[:3, 0] * width / 2 - matrix[:3, 1] * height / 2,
                        target + matrix[:3, 0] * width / 2 + matrix[:3, 1] * height / 2,
                        target - matrix[:3, 0] * width / 2 + matrix[:3, 1] * height / 2,
                    ]
                )
                geometry = np.vstack(
                    (camera, target, corners, target + normal * max(width, height) / 4)
                )
                screen = project(geometry)
                selected = item["view_id"] == self.selected_view_id
                color = QColor("#65d98b" if item["accepted"] else "#ff5d67")
                pen = QPen(color, 3.0 if selected else 1.1)
                painter.setPen(pen)
                if self.show_frusta:
                    for corner in range(2, 6):
                        painter.drawLine(
                            int(screen[0, 0]),
                            int(screen[0, 1]),
                            int(screen[corner, 0]),
                            int(screen[corner, 1]),
                        )
                    for first, second in ((2, 3), (3, 4), (4, 5), (5, 2)):
                        painter.drawLine(
                            int(screen[first, 0]),
                            int(screen[first, 1]),
                            int(screen[second, 0]),
                            int(screen[second, 1]),
                        )
                    painter.drawEllipse(int(screen[0, 0]) - 3, int(screen[0, 1]) - 3, 6, 6)
                if self.show_normals:
                    painter.setPen(QPen(QColor("#ffd166"), 1.5))
                    painter.drawLine(
                        int(screen[1, 0]),
                        int(screen[1, 1]),
                        int(screen[6, 0]),
                        int(screen[6, 1]),
                    )
            painter.setPen(QColor("#c9d3df"))
            painter.drawText(14, self.height() - 14, "左键拖动旋转 · 滚轮缩放 · 表格选择高亮")

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                self.last_position = event.position()

        def mouseMoveEvent(self, event) -> None:
            if self.last_position is None:
                return
            delta = event.position() - self.last_position
            self.last_position = event.position()
            self.yaw += delta.x() * 0.008
            self.pitch = float(np.clip(self.pitch + delta.y() * 0.008, -1.5, 1.5))
            self.update()

        def mouseReleaseEvent(self, _event) -> None:
            self.last_position = None

        def wheelEvent(self, event) -> None:
            self.zoom = float(
                np.clip(self.zoom * np.exp(event.angleDelta().y() / 1200.0), 0.2, 8.0)
            )
            self.update()

    application = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle("BiBladeFusion 精扫视点离线验收（禁止运动）")
    central = QWidget()
    layout = QVBoxLayout(central)
    status = QLabel(
        ("几何验收：通过" if payload["geometry_passed"] else "几何验收：失败")
        + "　|　机械臂可行性：未验证　|　motion_authorized: false"
    )
    status.setStyleSheet(
        "padding: 9px; font-weight: 600; color: "
        + ("#73e59a" if payload["geometry_passed"] else "#ff747d")
        + "; background: #1c232d;"
    )
    status.setToolTip("\n".join((*payload["global_reasons"], *payload["warnings"])))
    layout.addWidget(status)
    toolbar = QHBoxLayout()
    side_combo = QComboBox()
    side_combo.addItems(("all", "front", "back"))
    region_combo = QComboBox()
    region_combo.addItems(("all", *REGION_COLORS.keys()))
    frusta_box = QCheckBox("相机视锥")
    frusta_box.setChecked(True)
    normal_box = QCheckBox("分区法向")
    normal_box.setChecked(True)
    reset_button = QPushButton("重置视角")
    toolbar.addWidget(QLabel("侧面"))
    toolbar.addWidget(side_combo)
    toolbar.addWidget(QLabel("区域"))
    toolbar.addWidget(region_combo)
    toolbar.addWidget(frusta_box)
    toolbar.addWidget(normal_box)
    toolbar.addStretch(1)
    toolbar.addWidget(reset_button)
    layout.addLayout(toolbar)
    splitter = QSplitter()
    canvas = OrbitCanvas()
    table = QTableWidget(len(views), 7)
    table.setHorizontalHeaderLabels(("状态", "侧面", "区域", "距离/m", "投影", "可见", "视点ID"))
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    for row, item in enumerate(views):
        values = (
            "通过" if item["accepted"] else "拒绝",
            item["side"],
            item["region"],
            f"{item['standoff_distance_m']:.3f}",
            f"{item['projection_fraction']:.3f}",
            f"{item['visibility_fraction']:.3f}",
            item["view_id"],
        )
        for column, value in enumerate(values):
            cell = QTableWidgetItem(value)
            if column == 0:
                cell.setForeground(QColor("#55c77c" if item["accepted"] else "#f05f69"))
            table.setItem(row, column, cell)
    table.resizeColumnsToContents()
    splitter.addWidget(canvas)
    splitter.addWidget(table)
    splitter.setStretchFactor(0, 4)
    splitter.setStretchFactor(1, 2)
    layout.addWidget(splitter)
    details = QLabel("选择表格中的视点以查看验收结论。")
    details.setWordWrap(True)
    details.setStyleSheet("padding: 7px; color: #c9d3df; background: #161c24;")
    layout.addWidget(details)

    def apply_filters() -> None:
        canvas.side_filter = side_combo.currentText()
        canvas.region_filter = region_combo.currentText()
        for row, item in enumerate(views):
            visible = (canvas.side_filter == "all" or item["side"] == canvas.side_filter) and (
                canvas.region_filter == "all" or item["region"] == canvas.region_filter
            )
            table.setRowHidden(row, not visible)
        canvas.update()

    side_combo.currentTextChanged.connect(apply_filters)
    region_combo.currentTextChanged.connect(apply_filters)
    frusta_box.toggled.connect(
        lambda checked: (setattr(canvas, "show_frusta", checked), canvas.update())
    )
    normal_box.toggled.connect(
        lambda checked: (setattr(canvas, "show_normals", checked), canvas.update())
    )
    reset_button.clicked.connect(canvas.reset_view)

    def select_row(row: int, _column: int) -> None:
        canvas.selected_view_id = str(views[row]["view_id"])
        canvas.update()
        reasons = views[row]["reasons"]
        conclusion = "；".join(reasons) if reasons else "通过全部几何门槛"
        details.setText(
            f"{views[row]['view_id']} | {views[row]['side']}/{views[row]['region']} | "
            f"距离 {views[row]['standoff_distance_m']:.3f} m | {conclusion}"
        )

    table.cellClicked.connect(select_row)
    window.setCentralWidget(central)
    window.resize(1380, 820)
    window.show()
    return int(application.exec())
