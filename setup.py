from setuptools import setup, find_packages

setup(
    name="kanditioned",
    version="1.0.3",
    author="cats-marin",
    description="Fast, Conditioned KAN",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cats-marin/KANditioned",
    packages=find_packages(),
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["numpy", "matplotlib", "torch"],
    license="Apache-2.0",
    project_urls={
        "Source": "https://github.com/cats-marin/KANditioned",
        "Bug Tracker": "https://github.com/cats-marin/KANditioned/issues",
    },
    author_email="willbui256@gmail.com"
)
