"""Offline Qt3D inspection of the active ES68 + D435i collision assembly.

The module-level helpers deliberately avoid importing Qt or any device backend so the
kinematic composition can be regression-tested on headless machines.  The GUI is a
viewer only: it has no robot address, driver, permit, or motion control.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    Es68D435iCollisionTemplate,
)
from biblade_fusion.robotics.cs68_model import CS68_JOINT_NAMES
from biblade_fusion.robotics.es68_model import Es68KinematicModel

Matrix4 = NDArray[np.float64]

JOINT_LABELS_ZH: tuple[str, ...] = (
    "J1 肩部回转",
    "J2 肩部俯仰",
    "J3 肘部",
    "J4 腕部 1",
    "J5 腕部 2",
    "J6 腕部 3",
)

LINK_COLORS: Mapping[str, str] = {
    "base_link_inertia": "#89939f",
    "shoulder_link": "#3f83d5",
    "upperarm_link": "#36a7bf",
    "forearm_link": "#45a66b",
    "wrist_1_link": "#d6a13d",
    "wrist_2_link": "#d87942",
    "wrist_3_link": "#a66bd4",
    "d435i_collision_link": "#e45c65",
}


def parse_joint_degrees(value: str | Sequence[float]) -> tuple[float, ...]:
    """Parse six controller joint angles in degrees and return radians."""

    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        parsed = tuple(float(item) for item in tokens)
    else:
        parsed = tuple(float(item) for item in value)
    if len(parsed) != len(CS68_JOINT_NAMES) or not np.isfinite(parsed).all():
        raise ValueError("joint degrees must be a finite six-vector")
    return tuple(math.radians(item) for item in parsed)


def _pose_matrix(
    xyz_m: Sequence[float],
    rpy_rad: Sequence[float],
) -> Matrix4:
    roll, pitch, yaw = (float(item) for item in rpy_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    transform[:3, 3] = np.asarray(xyz_m, dtype=np.float64)
    return transform


def assembly_mesh_transforms(
    template: Es68D435iCollisionTemplate,
    joint_positions_rad: Sequence[float],
    *,
    kinematic_model: Es68KinematicModel | None = None,
) -> dict[str, Matrix4]:
    """Return ``base_T_raw_STL`` for all eight meshes at one controller pose.

    Mesh scale is applied last in the mesh-local frame.  Origins are already expressed
    in metres and therefore must not be scaled when an STL uses millimetres.
    """

    model = kinematic_model or Es68KinematicModel.from_resources()
    joints = tuple(float(item) for item in joint_positions_rad)
    if len(joints) != len(CS68_JOINT_NAMES) or not np.isfinite(joints).all():
        raise ValueError("joint positions must be a finite six-vector")
    base_t_link = model.link_transforms(joints)
    scale = np.diag((template.mesh_scale, template.mesh_scale, template.mesh_scale, 1.0))
    transforms: dict[str, Matrix4] = {}
    for spec in template.links:
        matrix = (
            base_t_link[spec.link_name]
            @ _pose_matrix(spec.origin_xyz_m, spec.origin_rpy_rad)
            @ scale
        )
        matrix.setflags(write=False)
        transforms[spec.link_name] = matrix

    attachment = template.attachment
    base_t_flange = model.base_t_flange(joints).matrix
    matrix = (
        base_t_flange
        @ _pose_matrix(attachment.joint_xyz_m, attachment.joint_rpy_rad)
        @ _pose_matrix(attachment.origin_xyz_m, attachment.origin_rpy_rad)
        @ scale
    )
    matrix.setflags(write=False)
    transforms[attachment.link_name] = matrix
    return transforms


def validate_joint_positions(
    joint_positions_rad: Sequence[float],
    kinematic_model: Es68KinematicModel,
) -> tuple[float, ...]:
    """Validate a controller pose against the same ES68 limits shown by the GUI."""

    joints = tuple(float(item) for item in joint_positions_rad)
    if len(joints) != len(CS68_JOINT_NAMES) or not np.isfinite(joints).all():
        raise ValueError("joint positions must be a finite six-vector")
    for index, (value, limits) in enumerate(
        zip(joints, kinematic_model.joint_limit_pairs(), strict=True),
        start=1,
    ):
        lower, upper = limits
        if value < lower or value > upper:
            raise ValueError(
                f"J{index}={math.degrees(value):.3f} deg is outside "
                f"[{math.degrees(lower):.3f}, {math.degrees(upper):.3f}] deg"
            )
    return joints


def assembly_mesh_paths(
    template: Es68D435iCollisionTemplate,
) -> dict[str, Path]:
    """Return the exact collision asset selected for every displayed link."""

    return {
        spec.link_name: spec.mesh_path
        for spec in (*template.links, template.attachment)
    }


def launch_es68_d435i_model_gui(
    *,
    resources: Es68D435iCollisionResources | None = None,
    joint_zero_offsets_rad: Sequence[float] = (),
    initial_joint_positions_rad: Sequence[float] = (0.0,) * 6,
) -> int:
    """Open the offline STL assembly inspector without touching physical hardware."""

    # PySide6 remains optional and is imported only at the GUI boundary.
    from PySide6 import Qt3DCore, Qt3DExtras, Qt3DRender
    from PySide6.QtCore import QSignalBlocker, Qt, QUrl
    from PySide6.QtGui import QColor, QMatrix4x4, QVector3D
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSlider,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    core3d = getattr(Qt3DCore, "Qt3DCore", Qt3DCore)
    extras3d = getattr(Qt3DExtras, "Qt3DExtras", Qt3DExtras)
    render3d = getattr(Qt3DRender, "Qt3DRender", Qt3DRender)

    resolved = resources or Es68D435iCollisionResources.packaged_template()
    template = resolved.load_active()
    offsets = tuple(float(item) for item in joint_zero_offsets_rad)
    model = Es68KinematicModel.from_resources(joint_zero_offsets_rad=offsets)
    joints = list(validate_joint_positions(initial_joint_positions_rad, model))

    application = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle("BiBladeFusion ES68+D435i STL 装配检查（完全离线）")

    central = QWidget()
    outer = QVBoxLayout(central)
    banner = QLabel(
        "离线 STL 装配预览　|　不连接机械臂或相机　|　不发送运动命令　|　"
        "motion_authorized: false"
    )
    banner.setStyleSheet(
        "padding: 10px; font-weight: 700; color: #ffd166; background: #202833;"
    )
    outer.addWidget(banner)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    view = extras3d.Qt3DWindow()
    view.defaultFrameGraph().setClearColor(QColor("#10151c"))
    container = QWidget.createWindowContainer(view)
    container.setMinimumSize(760, 620)
    splitter.addWidget(container)

    root_entity = core3d.QEntity()
    view.setRootEntity(root_entity)
    key_light_entity = core3d.QEntity(root_entity)
    key_light = render3d.QDirectionalLight(key_light_entity)
    key_light.setWorldDirection(QVector3D(-1.0, 0.6, -1.0))
    key_light.setColor(QColor("#ffffff"))
    key_light.setIntensity(1.0)
    key_light_entity.addComponent(key_light)

    fill_light_entity = core3d.QEntity(root_entity)
    fill_light = render3d.QDirectionalLight(fill_light_entity)
    fill_light.setWorldDirection(QVector3D(0.5, -1.0, -0.35))
    fill_light.setColor(QColor("#9fc7ff"))
    fill_light.setIntensity(0.45)
    fill_light_entity.addComponent(fill_light)
    camera = view.camera()
    camera.lens().setPerspectiveProjection(42.0, 16.0 / 9.0, 0.01, 10.0)

    def reset_camera() -> None:
        camera.setPosition(QVector3D(1.15, -1.15, 0.85))
        camera.setViewCenter(QVector3D(-0.28, 0.0, 0.28))
        camera.setUpVector(QVector3D(0.0, 0.0, 1.0))

    reset_camera()
    orbit = extras3d.QOrbitCameraController(root_entity)
    orbit.setCamera(camera)
    orbit.setLinearSpeed(0.8)
    orbit.setLookSpeed(180.0)

    mesh_paths = assembly_mesh_paths(template)
    mesh_entities: dict[str, object] = {}
    mesh_transforms: dict[str, object] = {}
    mesh_status: dict[str, str] = {name: "等待加载" for name in mesh_paths}
    status_table: QTableWidget | None = None

    def refresh_status_table() -> None:
        if status_table is None:
            return
        for row, name in enumerate(mesh_paths):
            status_table.item(row, 4).setText(mesh_status[name])

    for name, path in mesh_paths.items():
        entity = core3d.QEntity(root_entity)
        geometry = render3d.QMesh(entity)
        material = extras3d.QPhongMaterial(entity)
        material.setDiffuse(QColor(LINK_COLORS.get(name, "#aab4bf")))
        material.setSpecular(QColor("#e8edf2"))
        material.setShininess(28.0)
        transform = core3d.QTransform(entity)
        entity.addComponent(geometry)
        entity.addComponent(material)
        entity.addComponent(transform)
        mesh_entities[name] = entity
        mesh_transforms[name] = transform

        def record_status(value, *, link_name: str = name) -> None:
            status_name = getattr(value, "name", str(value))
            mesh_status[link_name] = {
                "None_": "等待加载",
                "Loading": "加载中",
                "Ready": "已加载",
                "Error": "加载失败",
            }.get(status_name, status_name)
            refresh_status_table()

        geometry.statusChanged.connect(record_status)
        geometry.setSource(QUrl.fromLocalFile(str(path)))

    panel = QWidget()
    panel_layout = QVBoxLayout(panel)
    model_label = QLabel(
        f"模型：{template.model_id}\n"
        f"单位：{template.mesh_units}　STL：{len(mesh_paths)}　"
        f"最小间隙策略：{template.minimum_clearance_m * 1000:.1f} mm\n"
        "此窗口只检查装配与关节联动；CLEAR/BLOCKED 路径验收需单独运行。"
    )
    model_label.setWordWrap(True)
    model_label.setStyleSheet("padding: 8px; color: #d8e0e8; background: #18202a;")
    panel_layout.addWidget(model_label)

    button_row = QHBoxLayout()
    zero_button = QPushButton("六轴归零")
    inspect_button = QPushButton("展示姿态")
    camera_button = QPushButton("重置视角")
    button_row.addWidget(zero_button)
    button_row.addWidget(inspect_button)
    button_row.addWidget(camera_button)
    panel_layout.addLayout(button_row)

    joint_group = QGroupBox("控制器关节角（仅更新离线模型）")
    joint_form = QFormLayout(joint_group)
    sliders: list[QSlider] = []
    spins: list[QDoubleSpinBox] = []
    for index, (label, limits) in enumerate(
        zip(JOINT_LABELS_ZH, model.joint_limit_pairs(), strict=True)
    ):
        minimum_deg, maximum_deg = (math.degrees(item) for item in limits)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(round(minimum_deg * 10), round(maximum_deg * 10))
        spin = QDoubleSpinBox()
        spin.setRange(minimum_deg, maximum_deg)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix("°")
        slider.setValue(round(math.degrees(joints[index]) * 10))
        spin.setValue(math.degrees(joints[index]))
        row_layout.addWidget(slider, 1)
        row_layout.addWidget(spin)
        joint_form.addRow(label, row)
        sliders.append(slider)
        spins.append(spin)
    panel_layout.addWidget(joint_group)

    mesh_group = QGroupBox("STL 图层")
    mesh_form = QFormLayout(mesh_group)
    for name, path in mesh_paths.items():
        checkbox = QCheckBox(path.name)
        checkbox.setChecked(True)
        checkbox.setStyleSheet(f"color: {LINK_COLORS.get(name, '#d8e0e8')};")
        checkbox.setToolTip(str(path))
        checkbox.toggled.connect(mesh_entities[name].setEnabled)
        mesh_form.addRow(name, checkbox)
    panel_layout.addWidget(mesh_group)

    status_table = QTableWidget(len(mesh_paths), 5)
    status_table.setHorizontalHeaderLabels(("Link", "X/m", "Y/m", "Z/m", "STL状态"))
    status_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    status_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    for row, name in enumerate(mesh_paths):
        for column, text in enumerate((name, "0", "0", "0", mesh_status[name])):
            status_table.setItem(row, column, QTableWidgetItem(text))
    status_table.resizeColumnsToContents()
    status_table.setMinimumHeight(260)
    panel_layout.addWidget(status_table)

    note = QLabel(
        "鼠标左键拖动：环绕观察　|　滚轮：缩放　|　右侧复选框：单独检查连杆\n"
        "重点检查 wrist3、安装件与 D435i 的方向及包络；本工具不替代真机安全验证。"
    )
    note.setWordWrap(True)
    note.setStyleSheet("padding: 8px; color: #aebbc8;")
    panel_layout.addWidget(note)
    panel_layout.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel)
    scroll.setMinimumWidth(500)
    splitter.addWidget(scroll)
    splitter.setStretchFactor(0, 4)
    splitter.setStretchFactor(1, 2)
    outer.addWidget(splitter)

    def qt_matrix(matrix: Matrix4) -> QMatrix4x4:
        return QMatrix4x4(*(float(item) for item in matrix.reshape(-1)))

    def update_scene() -> None:
        placements = assembly_mesh_transforms(
            template,
            joints,
            kinematic_model=model,
        )
        for row, (name, matrix) in enumerate(placements.items()):
            mesh_transforms[name].setMatrix(qt_matrix(matrix))
            for column, value in enumerate(matrix[:3, 3], start=1):
                status_table.item(row, column).setText(f"{float(value):.4f}")

    def slider_changed(index: int, value: int) -> None:
        degrees = value / 10.0
        with QSignalBlocker(spins[index]):
            spins[index].setValue(degrees)
        joints[index] = math.radians(degrees)
        update_scene()

    def spin_changed(index: int, value: float) -> None:
        with QSignalBlocker(sliders[index]):
            sliders[index].setValue(round(value * 10))
        joints[index] = math.radians(value)
        update_scene()

    for index, slider in enumerate(sliders):
        slider.valueChanged.connect(
            lambda value, joint_index=index: slider_changed(joint_index, value)
        )
    for index, spin in enumerate(spins):
        spin.valueChanged.connect(
            lambda value, joint_index=index: spin_changed(joint_index, value)
        )

    def apply_pose(values_deg: Sequence[float]) -> None:
        for index, degrees in enumerate(values_deg):
            with QSignalBlocker(sliders[index]), QSignalBlocker(spins[index]):
                sliders[index].setValue(round(float(degrees) * 10))
                spins[index].setValue(float(degrees))
            joints[index] = math.radians(float(degrees))
        update_scene()

    zero_button.clicked.connect(lambda: apply_pose((0.0,) * 6))
    inspect_button.clicked.connect(lambda: apply_pose((0.0, -60.0, 90.0, -60.0, -90.0, 0.0)))
    camera_button.clicked.connect(reset_camera)
    update_scene()

    window.setCentralWidget(central)
    window.resize(1500, 900)
    window.show()
    return int(application.exec())
