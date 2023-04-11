from ast import Assert
import logging

class Registry:

    def __init__(self, registry_name) -> None:
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value{value} of Registry is not callable!")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class"""

        def add(key, value):
            self[key] = value
            return value
        
        if callable(target):
            return add(None, target)
        
        return lambda x: add(target, x)

    def __getitem__(self, key):
        if key not in self._dict.keys():
            raise Exception(f"{key} is not offered now!")
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict
    
    @property
    def modules(self):
        """key"""
        return self._dict.keys()
    
    @property
    def name(self):
        return self._name



    