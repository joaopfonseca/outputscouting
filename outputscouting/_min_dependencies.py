"""All minimum dependencies for outputscouting."""

import argparse

NUMPY_MIN_VERSION = "1.20.0"
PANDAS_MIN_VERSION = "2.1.0"
SKLEARN_MIN_VERSION = "1.2.0"
TORCH_MIN_VERSION = "2.4.1"
MATPLOTLIB_MIN_VERSION = "3.9.2"
SEABORN_MIN_VERSION = "0.13.2"

# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": (NUMPY_MIN_VERSION, "install"),
    "pandas": (PANDAS_MIN_VERSION, "install"),
    "scikit-learn": (SKLEARN_MIN_VERSION, "install"),
    "torch": (TORCH_MIN_VERSION, "install"),
    "matplotlib": (MATPLOTLIB_MIN_VERSION, "install"),
    "seaborn": (SEABORN_MIN_VERSION, "install"),
    "numpydoc": ("1.0.0", "doc, tests"),
    "sphinx": ("8.0.2", "doc"),
    "sphinx-book-theme": ("1.1.3", "doc"),
    "recommonmark": ("0.7.1", "doc"),
    "sphinx-markdown-tables": ("0.0.15", "doc"),
    "sphinx-copybutton": ("0.4.0", "doc"),
    "sphinx-gallery": ("0.17.1", "doc"),
    "ipykernel": ("6.29.5", "doc"),
    "pandoc": ("2.4", "doc"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "doc", "examples", "tests", "all"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))
    tag_to_packages["all"].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
