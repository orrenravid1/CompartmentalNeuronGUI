from compneurovis.core import (
    AttributeRef,
    BoolValueSpec,
    ChoiceValueSpec,
    ControlPresentationSpec,
    ControlSpec,
    ScalarValueSpec,
    SeriesSpec,
    XYValueSpec,
)


class _Child:
    def __init__(self) -> None:
        self.value = 1.0


class _Root:
    def __init__(self) -> None:
        self.child = _Child()


def test_attribute_ref_reads_and_writes_nested_attribute():
    root = _Root()
    ref = AttributeRef("child", "value")

    assert ref.read(root) == 1.0
    ref.write(root, 2.5)
    assert root.child.value == 2.5


def test_control_spec_can_carry_attribute_target():
    spec = ControlSpec(
        id="gain",
        label="Gain",
        value_spec=ScalarValueSpec(default=1.0, min=0.0, max=2.0, value_type="float"),
        presentation=ControlPresentationSpec(kind="slider", steps=100),
        send_to_session=True,
        target=AttributeRef("child", "value"),
    )

    assert spec.target == AttributeRef("child", "value")


def test_control_spec_default_value_comes_from_value_spec():
    scalar = ControlSpec(id="gain", label="Gain", value_spec=ScalarValueSpec(default=1.0))
    choice = ControlSpec(id="mode", label="Mode", value_spec=ChoiceValueSpec(default="a", options=("a", "b")))
    flag = ControlSpec(id="enabled", label="Enabled", value_spec=BoolValueSpec(default=True))
    xy = ControlSpec(id="position", label="Position", value_spec=XYValueSpec(default={"x": 0.2, "y": 0.8}))

    assert scalar.default_value() == 1.0
    assert choice.default_value() == "a"
    assert flag.default_value() is True
    assert xy.default_value() == {"x": 0.2, "y": 0.8}
    assert xy.default_value() is not xy.value_spec.default


def test_series_spec_reads_from_attribute_ref():
    root = _Root()
    series = SeriesSpec("child_value", "Child value", source=AttributeRef("child", "value"), color=(1, 2, 3))

    assert series.source.read(root) == 1.0
    assert series.color == (1, 2, 3)
