from __future__ import annotations

import jax
import xarray as xr

import scm.physics.utils as phys_utils
from scm.config import load_namelist
from scm.examples.wangara import get_wangara_day33
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate

if __name__ == "__main__":
    ## Run standard Wangara simulation
    cfg = load_namelist("namelist_cn.yaml")
    sim = get_wangara_day33(Nz=100)
    model = init_model(sim, cfg)

    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out, sim)

    ## Now obtain Cn2 from postprocessing
    # Compute pressure
    p_s = 1000.0  # hPa, assumed surface pressure
    p_fn = jax.vmap(phys_utils.thermo.p_rho_from_th, in_axes=(0, 0, None, None))
    p, _ = p_fn(
        out.state_traj.th,
        out.state_traj.qv,
        sim.grid.z,
        p_s,
    )
    p = xr.DataArray(p, dims=["time", "z"])

    # Compute absolute temperature
    tk = phys_utils.thermo.th_to_tk(th=ds["th"], p_hPa=p)

    # Average CT2 to full-levels
    ct2 = ds["ct2"].drop_vars("zh")
    ct2 = (ct2.isel(zh=slice(0, -1)) + ct2.isel(zh=slice(1, None))) / 2
    ct2 = ct2.rename({"zh": "z"})

    # Compute Cn2 without humidity correction
    cn2_th = phys_utils.ot.cn2_th(ct2=ct2, p=p, tk=tk, th=ds["th"], bowen=None)
    cn2_tk = phys_utils.ot.cn2_tk(ct2=ct2, p=p, tk=tk, bowen=None)

    # Save everything in dataset
    ds["p"] = p
    ds["cn2_th"] = cn2_th
    ds["cn2_tk"] = cn2_tk
    ds.to_netcdf("out.nc")
