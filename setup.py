from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="stl-pinn-processor",
    version="1.0.0",
    author="Your Organization",
    author_email="contact@yourorg.com",
    description="Production-grade STL processing with Physics-Informed Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/stl-pinn-processor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "isort", "mypy", "flake8"],
        "docs": ["mkdocs", "mkdocs-material"],
        "gpu": ["torch[cuda]", "torchvision[cuda]"],
    },
    entry_points={
        "console_scripts": [
            "stl-pinn=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.toml"],
    },
)
