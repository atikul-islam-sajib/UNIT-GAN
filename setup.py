from setuptools import setup, find_packages

setup(
    name="UNIT-GAN",
    version="0.0.1",
    description="A deep learning project for segmentation task using UNIT-GAN".title(),
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    url="https://github.com/atikul-islam-sajib/UNIT-GAN",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="UNIT-GAN, Deep Learning: GAN",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/UNIT-GAN/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/UNIT-GAN",
        "Source Code": "https://github.com/atikul-islam-sajib/UNIT-GAN",
    },
)