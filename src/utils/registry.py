"""Lightweight registry and config builder utilities."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Iterable, Optional, Union


class Registry:
    """Map string names in config files to Python classes/functions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._module_dict: Dict[str, Any] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __repr__(self) -> str:
        registered = ", ".join(sorted(self._module_dict)) or "<empty>"
        return f"Registry(name={self.name}, items=[{registered}])"

    @property
    def module_dict(self) -> Dict[str, Any]:
        return self._module_dict

    def get(self, key: str) -> Any:
        return self._module_dict.get(key)

    def register_module(
        self,
        module: Optional[Any] = None,
        name: Optional[Union[str, Iterable[str]]] = None,
        force: bool = False,
    ) -> Callable[[Any], Any]:
        """Register a module class or function."""

        def _register(obj: Any) -> Any:
            names = name
            if names is None:
                names = [obj.__name__]
            elif isinstance(names, str):
                names = [names]
            else:
                names = list(names)

            for module_name in names:
                if not force and module_name in self._module_dict:
                    raise KeyError(
                        f"{module_name!r} is already registered in {self.name}."
                    )
                self._module_dict[module_name] = obj
            return obj

        if module is not None:
            return _register(module)
        return _register


def build_from_cfg(
    cfg: Optional[Dict[str, Any]],
    registry: Registry,
    default_args: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build an object from a config dict containing a ``type`` key."""

    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)!r}.")
    if "type" not in cfg:
        raise KeyError(f"cfg for registry {registry.name} must contain key 'type'.")

    args = copy.deepcopy(cfg)
    obj_type = args.pop("type")
    if default_args:
        for key, value in default_args.items():
            args.setdefault(key, value)

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            choices = ", ".join(sorted(registry.module_dict)) or "<empty>"
            raise KeyError(
                f"{obj_type!r} is not registered in {registry.name}. "
                f"Available: {choices}"
            )
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError("cfg['type'] must be a string or callable.")

    return obj_cls(**args)


DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")

MODELS = Registry("model")
BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")
LOSSES = Registry("loss")
DECODERS = Registry("decoder")

METRICS = Registry("metric")
VISUALIZERS = Registry("visualizer")
