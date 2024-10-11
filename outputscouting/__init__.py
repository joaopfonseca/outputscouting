"""
TODO: Add library description here
"""

import sys

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of mlresearch when
    # the binaries are not built
    # mypy error: Cannot determine type of '__OUTPUTSCOUTING_SETUP__'
    __OUTPUTSCOUTING_SETUP__  # type: ignore
except NameError:
    __OUTPUTSCOUTING_SETUP__ = False

if __OUTPUTSCOUTING_SETUP__:
    sys.stderr.write("Partial import of imblearn during the build process.\n")
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet
else:
    from .base import OutputScouting
    from ._command import CentralCommand
    from ._scout import Scout
    from ._temp_setter import sample_from_pdf, AuxTemperatureSetter

    __all__ = [
        "OutputScouting",
        "CentralCommand",
        "Scout",
        "AuxTemperatureSetter",
        "sample_from_pdf",
    ]
