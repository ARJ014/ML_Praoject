from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file: str) -> List[str]:
    requirements = []
    with open(file) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="MlProject",
    version="0.0.1",
    author="Arjun Goel",
    # packages=find_packages(),
    # install_requires=get_requirements('requirements.txt')
)
