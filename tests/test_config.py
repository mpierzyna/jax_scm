from scm.config import Namelist, load_namelist


def test_load_from_file(tmp_path):
    # Create a temporary config file
    cfg_file = tmp_path / "namelist.yaml"
    cfg_file.write_text("")  # create empty file. Hydra should fill in defaults.

    # Load config
    cfg = load_namelist(f=cfg_file)
    assert isinstance(cfg, Namelist)
