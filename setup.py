from setuptools import setup, find_packages

setup(
    name="pinpointing",
    version="0.1",
    description="tool for searching 1D DTW distance upstream and downstream",
    author="Josh Friedlander",
    author_email="josh@kando.eco",
    packages=find_packages(),
    install_requires=[
        "networkx", "kando-env", "numpy", "pandas", "pytz", "scipy",
        "python-dotenv", "tslearn"
    ],
)
