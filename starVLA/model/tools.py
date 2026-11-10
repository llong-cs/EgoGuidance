"""Model-package utilities: the framework registry."""


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        """Decorator: register a builder function or class under ``key`` (re-registration overwrites)."""

        def decorator(framework_class):
            self._registry[key] = framework_class
            return framework_class

        return decorator

    def __getitem__(self, key):
        return self._registry[key]

    def list(self):
        """Return a ``{key: registered object}`` mapping of everything registered so far."""
        return {k: v for k, v in self._registry.items()}


FRAMEWORK_REGISTRY = Registry("frameworks")
