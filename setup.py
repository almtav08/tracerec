from setuptools import setup, find_packages

setup(
    name="tracerec",
    version="0.1.0",
    description="Librería Python para la creación de sistemas de recomendación basados en trazas de usuario.",
    long_description=open("README.md").read(),
    author="Alex Martínez",
    author_email="alemarti@uji.es",
    packages=find_packages(),
    install_requires=["torch>=2.0.0"],
    python_requires=">=3.7",
    url="",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
