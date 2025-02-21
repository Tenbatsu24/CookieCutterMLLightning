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
