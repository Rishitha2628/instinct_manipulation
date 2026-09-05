from setuptools import setup, find_packages

setup(
    name="intrinsic_core",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy>=1.20"],
    extras_require={"viz": ["matplotlib>=3.4"]},
    python_requires=">=3.9",
    description="n-step empowerment estimation for continuous systems.",
    license="MIT",
)
