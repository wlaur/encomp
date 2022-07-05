import os

try:
    import fire
except ImportError:
    print('This script requires the "fire" package: pip install fire')
    fire = None


def build_docs():

    cmds = [

        # exclude the encomp.tests subpackage
        'sphinx-apidoc -f -o docs/source encomp encomp/tests',
        'call docs/make clean',
        'call docs/make html'
    ]

    for n in cmds:
        os.system(n)


def local_install():

    os.system('pip install .')


def build():

    cmds = [
        'rmdir /s/q build',
        'rmdir /s/q dist',
        'python setup.py bdist_wheel',
    ]

    for n in cmds:
        os.system(n)


def pip_upload():

    cmds = [
        'twine upload dist/*'
    ]

    for n in cmds:
        os.system(n)


def main(task=None):

    if task is None:
        build_docs()
        local_install()

    elif task == 'build':
        build()

    elif task == 'pip':
        build()
        pip_upload()

    elif task == 'docs':
        build_docs()

    elif task == 'install':
        local_install()

    else:
        raise ValueError(f'Unknown task: {task}')


if __name__ == '__main__':
    fire.Fire(main)
