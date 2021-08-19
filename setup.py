import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.readlines()

setuptools.setup(
    name="deeplense_domain_adaptation",
    version="0.0.2",
    author="Marcos Tidball",
    author_email="marcostidball@gmail.com",
    description="A PyTorch-based collection of Unsupervised Domain Adaptation methods applied to strong gravitational lenses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zysymu/Domain-Adaptation-DeepLense",
    project_urls={
        "Bug Tracker": "https://github.com/zysymu/Domain-Adaptation-DeepLense/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "deeplense_domain_adaptation"},
    packages=setuptools.find_packages(where="deeplense_domain_adaptation"),
    install_requires=install_requires,
    python_requires=">=3.9.5",
)