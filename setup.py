import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="PyXCSAO",
    version="0.1",
    author="Marina Kounkel",
    author_email="marina.kounkel@vanderbilt.edu",
    description="Replicates functionality of IRAF XCSAO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkounkel/pyxcsao",
    packages=['pyxcsao'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        'pyxcsao': ['getha.pt','getli.pt'],
    },
    install_requires=['astropy','numpy','scipy','specutils','PyAstronomy','pytorch','torchvision'],
    python_requires='>=3.6',
)
