__version__ = '0.1.0'

from typeguard.importhook import install_import_hook

# installs the @typechecked decorator on all functions imported
# after this hook is executed
install_import_hook('encomp')
