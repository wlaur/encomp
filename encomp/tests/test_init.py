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
