__version__ = '0.1.1'

from encomp.settings import SETTINGS

# TODO: use env variables to disable this before importing encomp
# cannot import encomp.settings.SETTINGS and change this, since __init__.py
# is executed before the SETTINGS object can be modified
if SETTINGS.type_checking:

    from typeguard.importhook import install_import_hook

    # installs the @typechecked decorator on all functions imported
    # after this hook is executed
    install_import_hook('encomp')
