class LooseVersion(str):
    def __new__(cls, version):
        return str.__new__(cls, version)
