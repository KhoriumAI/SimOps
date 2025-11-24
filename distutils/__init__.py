# Minimal stub for distutils package to satisfy legacy imports
# Provides a LooseVersion class compatible with qtrangeslider's usage.
# This implementation simply inherits from str for basic comparison.

class LooseVersion(str):
    def __new__(cls, version):
        return str.__new__(cls, version)
