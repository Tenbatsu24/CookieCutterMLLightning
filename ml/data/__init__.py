from ml.util import STORE

TYPE_OF: str = "data"


@STORE.reg(TYPE_OF, "test")
class Test:
    pass
