from compneurovis.core import AttributeRef, ControlSpec, SeriesSpec


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
        "gain",
        "float",
        "Gain",
        1.0,
        min=0.0,
        max=2.0,
        steps=100,
        send_to_session=True,
        target=AttributeRef("child", "value"),
    )

    assert spec.target == AttributeRef("child", "value")


def test_series_spec_reads_from_attribute_ref():
    root = _Root()
    series = SeriesSpec("child_value", "Child value", source=AttributeRef("child", "value"), color=(1, 2, 3))

    assert series.source.read(root) == 1.0
    assert series.color == (1, 2, 3)
