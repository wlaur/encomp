from pathlib import Path

from ..context import quantity_format, silence_stdout, temp_dir, working_dir
from ..units import Quantity


def test_context() -> None:
    with silence_stdout():
        print("silenced")

    with quantity_format("~Lx"):
        qty = Quantity(1, "kPa")
        s = str(qty)

        # NOTE: inputs are cast to float or array[float]
        assert s == r"\SI[]{1.0}{\kilo\pascal}"

    s = str(Quantity(1, "kPa"))
    assert s == "1.0 kPa"


def test_quantity_format_aliases() -> None:
    # the default argument and every documented alias must be accepted
    with quantity_format():
        assert str(Quantity(1, "kPa")) == "1.0 kPa"

    for alias, spec in [("compact", "~P"), ("normal", "~P"), ("siunitx", "~Lx")]:
        with quantity_format(alias):
            assert str(Quantity(1, "kPa")) == f"{Quantity(1, 'kPa'):{spec}}"

    assert str(Quantity(1, "kPa")) == "1.0 kPa"


def test_temp_dir() -> None:
    orig_cwd = Path.cwd()

    with temp_dir() as temp:
        assert Path.cwd() == temp

        assert temp != orig_cwd

    assert not temp.is_dir()


def test_working_dir() -> None:
    with temp_dir():
        parent = Path.cwd()
        sub = parent / "sub"

        sub.mkdir(exist_ok=False)

        with working_dir(sub):
            assert Path.cwd().is_relative_to(parent)

    assert not sub.is_dir()
    assert not parent.is_dir()
