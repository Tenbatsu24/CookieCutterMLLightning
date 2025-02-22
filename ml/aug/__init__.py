from ml.util import STORE

TYPE_OF: str = "aug"


@STORE.reg(TYPE_OF, "test")
class AugTest:
    pass
