"""Deprecated alias: Topoformer was renamed to Tensegra.

``import topoformer.x`` and ``python -m topoformer.x`` resolve to the same
module objects as ``tensegra.x``, so frozen configs, launch receipts and
research tools that name the old package keep working.
"""
import importlib
import importlib.abc
import importlib.util
import sys

import tensegra as _tensegra
from tensegra import *  # noqa: F401,F403

_OLD, _NEW = __name__, 'tensegra'


def _target(name):
    return _NEW + name[len(_OLD):]


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _specs = {}

    def find_spec(self, name, path=None, target=None):
        if not name.startswith(_OLD + '.'):
            return None
        spec = importlib.util.find_spec(_target(name))
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            name, self, origin=spec.origin, is_package=spec.submodule_search_locations is not None)

    def create_module(self, spec):
        module = importlib.import_module(_target(spec.name))
        self._specs[spec.name] = module.__spec__
        return module

    # The import system overwrites __spec__ on the returned module; restore the real one.
    def exec_module(self, module):
        module.__spec__ = self._specs.pop(module.__spec__.name)

    # runpy (``python -m topoformer.x``) executes the target's code as __main__.
    def get_code(self, name):
        target = _target(name)
        return importlib.util.find_spec(target).loader.get_code(target)

    def is_package(self, name):
        return importlib.util.find_spec(_target(name)).submodule_search_locations is not None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

__path__ = []
