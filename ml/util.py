class SingletonRegistry:
    _instance = None
    _registry = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, type_of, name, cls):
        if type_of not in self._registry:
            self._registry[type_of] = {}
        self._registry[type_of][name] = cls

    def get_class(self, type_of, name):
        if type_of not in self._registry:
            raise ValueError(f"Unknown type {type_of}")
        if name not in self._registry[type_of]:
            raise ValueError(f"Unknown {type_of} {name}")
        return self._registry[type_of][name]

    def get_registry(self):
        return self._registry


registry = SingletonRegistry()


def register(type_of, name):
    """
    Register a class with the registry
    :param type_of: the top level type of the class. it will be the first key in the registry
    :param name: the name of the class. it will be the second key in the registry
    :return: the class
    """

    def inner(cls):
        registry.register(type_of, name, cls)
        return cls

    return inner


def get_class(type_of, name):
    """
    Get a class from the registry
    :param type_of: the top level type of the class
    :param name: the name of the class
    :return: class
    """
    return registry.get_class(type_of, name)
