from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from shared import FIXTURE_ROOT

from scm.config import LogLevel, load_namelist
from scm.examples.andren1994.andren1994 import get_andren1994
from scm.examples.gabls1 import get_gabls1
from scm.examples.wangara.wangara import get_wangara_day33
from scm.interfaces import Simulation
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate


class CaseSpec(NamedTuple):
    fixture_dir: str
    out_file: str
    namelist: str
    get_sim: Callable[[], Simulation]


CASES = {
    "gabls1_cn": CaseSpec(
        fixture_dir="gabls1",
        out_file="out_cn.nc",
        namelist="namelist_cn.yaml",
        get_sim=lambda: get_gabls1(Nz=64),
    ),
    "gabls1_ab2": CaseSpec(
        fixture_dir="gabls1",
        out_file="out_ab2.nc",
        namelist="namelist_ab2.yaml",
        get_sim=lambda: get_gabls1(Nz=64),
    ),
    "andren1994_cn": CaseSpec(
        fixture_dir="andren1994",
        out_file="out_cn.nc",
        namelist="namelist_cn.yaml",
        get_sim=lambda: get_andren1994(Nz=100),
    ),
    "andren1994_ab2": CaseSpec(
        fixture_dir="andren1994",
        out_file="out_ab2.nc",
        namelist="namelist_ab2.yaml",
        get_sim=lambda: get_andren1994(Nz=100),
    ),
    "wangara_cn": CaseSpec(
        fixture_dir="wangara",
        out_file="out_cn.nc",
        namelist="namelist_cn.yaml",
        get_sim=lambda: get_wangara_day33(Nz=100),
    ),
    "wangara_ab2": CaseSpec(
        fixture_dir="wangara",
        out_file="out_ab2.nc",
        namelist="namelist_ab2.yaml",
        get_sim=lambda: get_wangara_day33(Nz=100),
    ),
}


@pytest.mark.parametrize("case", list(CASES.keys()))
def test_e2e(case: str) -> None:
    """Test that the full simulation can be run and produces expected output."""
    spec = CASES[case]
    fixture_dir = FIXTURE_ROOT / spec.fixture_dir

    ds = xr.open_dataset(fixture_dir / spec.out_file)
    cfg = load_namelist(fixture_dir / spec.namelist)
    cfg.logging.level = LogLevel.SILENT

    with jax.enable_x64():
        sim = spec.get_sim()
        model = init_model(sim, cfg=cfg)
        out = simulate(model=model, sim=sim, cfg=cfg)
        ds_new = out_to_ds(out, sim=sim)

    for var in ds.data_vars:
        if "frc" in var:
            continue
        if "qv" in var:
            continue

        ref_mean = np.abs(ds[var].values).mean()
        # Skip variables that are zero (division by zero) or non-finite (e.g. Obukhov
        # length in neutral cases where L → ∞).
        if not np.isfinite(ref_mean) or ref_mean < 1e-10:
            continue

        rel_err = np.abs((ds[var].values - ds_new[var].values) / ref_mean).max()
        assert rel_err < 1e-4, f"Variable {var} differs between runs by more than 1e-4 relative error"


def test_split_sim() -> None:
    """Test that reference simulation output can be split, run separately, and concatenated back together to match the original output."""
    # Run full trajectory
    spec = CASES["gabls1_ab2"]
    fixture_dir = FIXTURE_ROOT / spec.fixture_dir

    cfg = load_namelist(fixture_dir / spec.namelist)
    cfg.logging.level = LogLevel.SILENT

    with jax.enable_x64():
        sim = spec.get_sim()
        model = init_model(sim, cfg=cfg)
        out_ref = simulate(model=model, sim=sim, cfg=cfg)

    out_1h = out_ref[out_ref.t_s % 3600 == 0]  # Select every hour
    out_1h = out_1h[1:-1]  # skip first and last hours

    # Create sub simulations
    sims = []
    for out_ in out_1h:
        sim_ = sim.update_init(
            new_t_start_s=int(out_.t_s),
            new_init=out_.state_traj,
        )
        sim_ = sim_.update(t_end_s=sim_.t_start_s + 3600)
        sims.append(sim_)

    # Run all sub sims
    with jax.enable_x64():
        model = init_model(sim, cfg=cfg)
        out_split = []
        for sim_ in sims:
            out_ = simulate(model=model, sim=sim_, cfg=cfg)
            out_ = out_[:-1]  # remove last point to avoid overlap
            out_split.append(out_)

        # Concat results
        out_concat = out_split[0]
        for out_ in out_split[1:]:
            out_concat = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), out_concat, out_)

        # Compare to reference
        out_ref = out_ref[out_ref.t_s >= 3600]  # skip first hour (spinup)
        out_ref = out_ref[:-1]  # skip last because removed in sim iter

        # Compare
        err = jax.tree.map(lambda x, x_: jnp.mean((x - x_) ** 2), out_concat, out_ref)
        err, _ = jax.tree.flatten(err)
        err = jnp.array(err)
        assert jnp.mean(err) < 1e-6
        assert jnp.max(err) < 1e-4
