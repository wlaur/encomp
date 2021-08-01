import os
import fire


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


def docker_build():

    os.system('docker build -t encomp . --no-cache')


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

    elif task == 'dbuild':
        build()
        docker_build()

    else:
        raise ValueError(f'Unknown task: {task}')


if __name__ == '__main__':
    fire.Fire(main)
