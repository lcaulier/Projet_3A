from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
        ]
    },
    description="Classification of Front/Ingredients/Nutritional images in OFF Database",
    author="lcaulier",
    license="MIT",
    entry_points={"console_scripts": ["monprojet=monprojet.cli:main"]},
)
