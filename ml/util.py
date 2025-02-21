class SingletonRegistry:
    _instance = None
    _registry = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, name, cls):
        self._registry[name] = cls

    def get_class(self, name):
        if name not in self._registry:
            raise ValueError(f"Unknown {name}")
        return self._registry[name]

    def get_registry(self):
        return self._registry

    def __str__(self):
        return f"\n".join([f"    {k}: {v}" for k, v in self._registry.items()])


class RegistryStore:
    def __init__(self):
        self._instances = {}

    def get_instance(self, type_of):
        if type_of not in self._instances:
            self._instances[type_of] = SingletonRegistry()
        return self._instances[type_of]

    def reg(self, type_of, name):
        """
        Register a class with the registry
        :param type_of: the top level type of the class. it will be the first key in the registry
        :param name: the name of the class. it will be the second key in the registry
        :return: the class
        """

        def inner(cls):
            register = self.get_instance(type_of)
            register.register(name, cls)
            return cls

        return inner

    def get(self, type_of, name):
        """
        Get a class from the registry
        :param type_of: the top level type of the class
        :param name: the name of the class
        :return: class
        """
        register = self.get_instance(type_of)
        return register.get_class(name)

    def __str__(self):
        return f"\nWarehouse:\n" + "\n".join([f"  {k}: {v}" for k, v in self._instances.items()])


STORE = RegistryStore()
