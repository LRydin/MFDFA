import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MFDFA",
    version="0.4",
    author="Leonardo Rydin Gorjao",
    author_email="leonardo.rydin@gmail.com",
    description="Multifractal Detrended Fluctuation Analysis in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LRydin/MFDFA",
    packages=setuptools.find_packages(),
    install_requires = ["numpy"],
    extras_require = {"EMD-signal": ["EMD-signal"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
)
