import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Mon-MHC",
    version="1.0.0",
    author="Marina Vallejo",
    author_email="marina.vallejo01@estudiant.upf.edu",
    description="Development of an MHC-I-peptide binding predictor using Monte Carlo simulations (Vallejo-Vallés et al. 2023)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    ],
)
