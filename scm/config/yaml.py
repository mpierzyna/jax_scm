"""Custom YAML serializers"""

from __future__ import annotations

import pathlib
from typing import Dict

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def path_representer(dumper: yaml.Dumper, path: pathlib.Path):
    """Represent ``pathlib.Path`` as string in yaml."""
    return dumper.represent_scalar(tag="!path", value=str(path))


def path_constructor(loader: yaml.loader, node) -> pathlib.Path:
    """Convert string back to ``pathlib.Path`` object."""
    return pathlib.Path(loader.construct_scalar(node))


def tuple_representer(dumper: yaml.Dumper, t: tuple):
    """Convert tuple to yaml list. Attention! Deserialisation will be list not tuple! Pydantic will fix that."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", t, flow_style=True)


# Register path representer and constructor for Posix and Windows to Dumper
yaml.add_representer(pathlib.Path, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.PosixPath, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.WindowsPath, path_representer, Dumper=Dumper)
yaml.add_constructor("!path", path_constructor, Loader=Loader)

# Register representer and constructor to convert tuple between Python and yaml.
yaml.add_representer(tuple, tuple_representer, Dumper=Dumper)


def yaml_to_dict(yaml_str: str) -> Dict:
    """Convert yaml string to dict."""
    if yaml_str == "":
        return {}
    return yaml.load(yaml_str, Loader=Loader)


def dict_to_yaml(d: Dict) -> str:
    """Convert dict to yaml string."""
    return yaml.dump(d, Dumper=Dumper)
