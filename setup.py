from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="medAId",
    version="0.1.0",
    description="Automated Machine Learning for medical use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeptuchMateusz/medAId",
    author="Zofia Kamińska, Karolina Dunal, Mateusz Deptuch",
    license="MIT",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
    python_requires='>=3.8',
    keywords=[
        "automated machine learning",
        "automl",
        "machine learning",
        "medical data",
    ],
    test_suite="tests",
    tests_require=["pytest", "unittest2"],
)
