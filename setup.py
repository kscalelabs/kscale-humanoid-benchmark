# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("benchmark/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()

requirements_dev = [
    "black",
    "darglint",
    "mypy",
    "pytest",
    "ruff",
]


with open("benchmark/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in benchmark/__init__.py"
version: str = version_re.group(1)


setup(
    name="benchmark",
    version=version,
    description="The benchmark project",
    author="K-Scale Labs",
    url="https://github.com/kscalelabs/kscale-humanoid-benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    packages=find_packages(),
)
