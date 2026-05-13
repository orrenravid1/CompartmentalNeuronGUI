import numpy as np
import pytest

from compneurovis.core import (
    AppSpec,
    ControlSpec,
    DataCatalog,
    Field,
    InteractionCatalog,
    LayoutCatalog,
    LayoutSpec,
    LinePlotViewSpec,
    ScalarValueSpec,
    ViewCatalog,
)


def _field() -> Field:
    return Field(
        id="voltage",
        values=np.zeros(2, dtype=np.float32),
        dims=("time",),
        coords={"time": np.array([0.0, 1.0], dtype=np.float32)},
    )


def test_app_spec_accepts_catalogs_and_exposes_legacy_views():
    field = _field()
    view = LinePlotViewSpec(id="trace", field_id=field.id)
    control = ControlSpec(id="gain", label="Gain", value_spec=ScalarValueSpec(default=1.0))

    app_spec = AppSpec(
        data=DataCatalog(fields={field.id: field}),
        view_catalog=ViewCatalog(views={view.id: view}),
        interactions=InteractionCatalog(controls={control.id: control}),
        layout_catalog=LayoutCatalog(layouts={"main": LayoutSpec(title="Main")}, active="main"),
        metadata={"source": "test"},
    )

    assert app_spec.data.fields == {field.id: field}
    assert app_spec.view_catalog.views == {view.id: view}
    assert app_spec.interactions.controls == {control.id: control}
    assert app_spec.active_layout().title == "Main"
    assert app_spec.metadata == {"source": "test"}

    assert app_spec.data.fields is app_spec.data.fields
    assert app_spec.view_catalog.views is app_spec.view_catalog.views
    assert app_spec.interactions.controls is app_spec.interactions.controls


def test_app_spec_keeps_flat_constructor_as_catalog_lowering():
    field = _field()
    view = LinePlotViewSpec(id="trace", field_id=field.id)

    app_spec = AppSpec(
        fields={field.id: field},
        geometries={},
        views={view.id: view},
        layout=LayoutSpec(title="Flat"),
    )

    assert app_spec.data.fields[field.id] is field
    assert app_spec.view_catalog.views[view.id] is view
    assert app_spec.active_layout()_catalog.active == "default"
    assert app_spec.active_layout().title == "Flat"


def test_app_spec_rejects_mixed_catalog_and_flat_inputs():
    field = _field()

    with pytest.raises(TypeError, match="data=DataCatalog"):
        AppSpec(data=DataCatalog(), fields={field.id: field})

    with pytest.raises(TypeError, match="view_catalog=ViewCatalog"):
        AppSpec(view_catalog=ViewCatalog(), views={})

    with pytest.raises(TypeError, match="interactions=InteractionCatalog"):
        AppSpec(interactions=InteractionCatalog(), controls={})

    with pytest.raises(TypeError, match="layout_catalog=LayoutCatalog"):
        AppSpec(layout_catalog=LayoutCatalog.single(), layout=LayoutSpec())


def test_layout_catalog_requires_active_layout_to_exist():
    with pytest.raises(ValueError, match="Active layout 'missing'"):
        LayoutCatalog(layouts={"main": LayoutSpec()}, active="missing")
