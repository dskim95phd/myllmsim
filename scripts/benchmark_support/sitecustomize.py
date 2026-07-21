"""Expose the mounted Chakra checkout without installing it into the image."""

import os
from pathlib import Path
import sys
import types


chakra_root = os.environ.get("CHAKRA_ROOT")
if chakra_root:
    root = Path(chakra_root)
    package_paths = {
        "chakra": root,
        "chakra.src": root / "src",
        "chakra.schema": root / "schema",
    }
    for package_name, package_path in package_paths.items():
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            sys.modules[package_name] = package
