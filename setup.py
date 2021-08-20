import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

#with open("requirements.txt", "r") as f:
#    install_requires = f.readlines()
#    install_requires = [x.strip() for x in install_requires]

setuptools.setup(
    name="deeplense_domain_adaptation",
    version="0.0.5",
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
    install_requires=["numpy==1.21.2",
                      "torch==1.9.0",
                      "e2cnn==0.1.9",
                      "torchvision==0.10.0",
                      "matplotlib==3.4.3",
                      "scikit-learn==0.24.2",
                      "scipy==1.7.1",
                      "seaborn==0.11.2"],
    python_requires=">=3.7.10",
)