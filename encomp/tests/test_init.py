import subprocess
import sys

import pytest


def test_top_level_import_is_clean() -> None:
    script = (
        "import sys\n"
        "import encomp\n"
        "print(hasattr(encomp, 'importlib'))\n"
        "print(hasattr(encomp, 'Q'))\n"
        "print(hasattr(encomp, 'Quantity'))\n"
        "print(hasattr(encomp, 'Fluid'))\n"
        "print('encomp.units' in sys.modules)\n"
        "print(encomp.__all__)"
    )

    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    assert result.stdout.splitlines() == ["False", "False", "False", "False", "False", "('__version__',)"]


def test_top_level_shortcuts_are_not_reexported() -> None:
    import encomp

    for name in ("Q", "Quantity", "Fluid", "Water", "HumidAir", "CONSTANTS"):
        with pytest.raises(AttributeError):
            getattr(encomp, name)

    with pytest.raises(ImportError):
        exec("from encomp import Q")


def test_coolprop_registers_unit_dtype_without_importing_units() -> None:
    script = (
        "import sys\n"
        "from encomp import coolprop\n"
        "import polars as pl\n"
        "print('encomp.units' in sys.modules)\n"
        "print(type(pl.DataFrame({'x': [1.0]}).schema['x']).__name__)\n"
        "print('encomp._polars_dtype' in sys.modules)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    assert result.stdout.splitlines() == ["False", "Float64", "True"]
