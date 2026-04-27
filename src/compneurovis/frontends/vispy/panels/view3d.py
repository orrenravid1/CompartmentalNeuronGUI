from __future__ import annotations

from PyQt6 import QtWidgets

from compneurovis.core.scene import PanelSpec
from compneurovis.frontends.vispy.view3d.viewport import Viewport3DPanel
from compneurovis.frontends.vispy.view3d.visuals import builtin_3d_visuals


class IndependentCanvas3DHostPanel(QtWidgets.QGroupBox):
    def __init__(self, *, panel: PanelSpec, title: str | None = None, on_entity_selected=None, parent=None):
        super().__init__(title or panel.view_ids[0], parent)
        self.panel_id = panel.id
        self.view_ids = panel.view_ids
        self.viewport = Viewport3DPanel(host_spec=panel, on_entity_selected=on_entity_selected)
        for key, visual in builtin_3d_visuals(self.viewport.view, panel_id=self.panel_id).items():
            self.viewport.mount_visual(key, visual)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(self.viewport)

    def clear(self) -> None:
        self.viewport.clear()

    def activate_visual(self, view_id: str, visual_key: str):
        if view_id != self.view_ids[0]:
            return None
        return self.viewport.activate_visual(visual_key)

    def visual(self, visual_key: str):
        return self.viewport.visual(visual_key)

    def set_background(self, color) -> None:
        self.viewport.canvas.bgcolor = color

    def commit(self) -> None:
        self.viewport.commit()
