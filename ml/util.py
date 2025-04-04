from importlib import import_module

MODEL_TYPE: str = "model"
AUG_TYPE: str = "aug"
DATA_TYPE: str = "data"
MIX_TYPE: str = "mix"

STORE_TYPES = [MODEL_TYPE, AUG_TYPE, DATA_TYPE, MIX_TYPE]


class InstanceRegistry:

    def __init__(self):
        self._registry = {}

    def register(self, name, cls):
        self._registry[name] = cls

    def get_class(self, name):
        if name not in self._registry:
            raise ValueError(f"Unknown {name}")
        return self._registry[name]

    def get_registry(self):
        return self._registry

    def __str__(self):
        return "\n" + "\n".join([f"    {k}: {v}" for k, v in self._registry.items()])


class RegistryStore:
    _instance = None
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RegistryStore, cls).__new__(cls)
        return cls._instance

    def get_instance(self, type_of):
        if type_of not in self._instances:
            self._instances[type_of] = InstanceRegistry()
        return self._instances[type_of]

    def register(self, type_of: str, name: str, cls):
        self.get_instance(type_of).register(name, cls)

    def reg(self, type_of: str, name: str):
        """
        Register a class with the registry
        :param type_of: the top level type of the class. it will be the first key in the registry
        :param name: the name of the class. it will be the second key in the registry
        :return: the class
        """

        def inner(cls):
            self.register(type_of, name, cls)
            return cls

        return inner

    def get(self, type_of: str, name: str) -> type:
        """
        Get a class from the registry
        :param type_of: the top level type of the class
        :param name: the name of the class
        :return: class
        """
        # check if there is a package with the name type_of. if it exists import * from it
        register = self.get_instance(type_of)
        return register.get_class(name)

    def __str__(self):
        return f"\n{self.__class__.__name__}:\n" + "\n".join(
            [f"  {k}: {v}" for k, v in self._instances.items()]
        )


STORE = RegistryStore()

## import all the modules in the store types such that they are registered
pkg = "ml"
for name in STORE_TYPES:
    mod = import_module(f".{name}", package=pkg)

del pkg, name, mod
