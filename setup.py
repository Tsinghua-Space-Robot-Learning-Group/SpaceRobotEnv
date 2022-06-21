from os.path import dirname, realpath
from setuptools import find_packages, setup

def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dirname(realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name="SpaceRobotEnv",
    version="0.0.1",
    install_requires=read_requirements_file('requirements.txt'),
    packages=find_packages(exclude=("image",)),
)