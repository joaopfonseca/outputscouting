"""All minimum dependencies for outputscouting."""

import argparse

NUMPY_MIN_VERSION = "1.20.0"
PANDAS_MIN_VERSION = "2.1.0"
SKLEARN_MIN_VERSION = "1.2.0"


# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": (NUMPY_MIN_VERSION, "install"),
    "pandas": (PANDAS_MIN_VERSION, "install"),
    "scikit-learn": (SKLEARN_MIN_VERSION, "install"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "docs", "examples", "tests", "all"]
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
