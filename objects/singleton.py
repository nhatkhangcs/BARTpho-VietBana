# class Singleton(type):
#     _instances = {}

#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]

from weakref import WeakValueDictionary

class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # This variable declaration is required to force a
            # strong reference on the instance.
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
