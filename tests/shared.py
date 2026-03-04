import os
import pathlib

FIXTURE_ROOT = pathlib.Path(os.path.dirname(__file__)) / "fixtures"
assert FIXTURE_ROOT.exists()
