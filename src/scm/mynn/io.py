import jax.numpy as jnp
import xarray as xr

from scm.forcing.interp import get_ts_interp_fn
from scm.grid import StaggeredGrid
from scm.interfaces import Forcing, Simulation
from scm.mo import MOSettings
from scm.mynn.interfaces import ProgVarsMYNN


def sim_from_ds(ds: xr.Dataset, **override) -> Simulation:
    """Load simulation from dataset.

    Note
    ----
    Forcing functions are restored by interpolating the saved time series. This may not exactly reproduce
    the original forcing, leading to error accumulation over time. Hence, the simulated output may deviate from the
    original. Simple tests (see tests/test_io.py) show that mean error is typically below 0.1%.
    """
    ## Grid
    H = ds["zh"][-1].item()
    Nz = ds.sizes["z"]
    grid = StaggeredGrid(H=H, Nz=Nz)

    ## Forcing
    time_s = jnp.array(ds["_t_s"].values)

    # Geostrophic wind
    ug = jnp.array(ds["frc_u_geo"].values)  # shape (time, z)
    vg = jnp.array(ds["frc_v_geo"].values)  # shape (time, z)

    # Surface heat forcing
    if "frc_th_s" in ds:
        frc_w_th_s = None
        frc_th_s = jnp.array(ds["frc_th_s"].values)  # shape (time,)
        frc_th_s = get_ts_interp_fn(time_s=time_s, data=frc_th_s)
    elif "frc_w_th_s" in ds:
        frc_w_th_s = jnp.array(ds["frc_w_th_s"].values)  # shape (time,)
        frc_w_th_s = get_ts_interp_fn(time_s=time_s, data=frc_w_th_s)
        frc_th_s = None
    else:
        raise ValueError("Dataset must contain either 'frc_th_s' or 'frc_w_th_s' for surface heat forcing")

    # Surface moisture forcing
    frc_w_qv_s = jnp.array(ds["frc_w_qv_s"].values)
    frc_w_qv_s = get_ts_interp_fn(time_s=time_s, data=frc_w_qv_s)

    # Coriolis parameter
    f_c = ds["frc_f_c"].item()
    dth_dz_top = ds["frc_dth_dz_top"].item()

    forcing = Forcing(
        u_geo=get_ts_interp_fn(time_s=time_s, data=ug),
        v_geo=get_ts_interp_fn(time_s=time_s, data=vg),
        f_c=f_c,
        th_s=frc_th_s,
        w_th_s=frc_w_th_s,
        w_qv_s=frc_w_qv_s,
        dth_dz_top=dth_dz_top,
    )

    ## MO settings
    # todo: why again did I want to override this?
    mo_settings = override.get("mo_settings")
    if mo_settings is None:
        mo_settings = MOSettings.deserialize(ds.attrs["mo_settings"])

    ## Reference temperature
    th_ref = ds.attrs["th_ref"]

    ## Initial conditions
    ds_init = ds.isel(time=0)
    u = jnp.array(ds_init["u"].values)
    v = jnp.array(ds_init["v"].values)
    th = jnp.array(ds_init["th"].values)
    qv = jnp.array(ds_init["qv"].values)
    qke = jnp.array(ds_init["qke"].values)
    init = ProgVarsMYNN(u=u, v=v, th=th, qke=qke, qv=qv)

    ## Meta data
    name = override.get("name", ds.attrs.get("name", "loaded_simulation"))
    t_start_s, t_end_s = ds["_t_s"].values[[0, -1]]

    return Simulation(
        name=name,
        grid=grid,
        init=init,
        forcing=forcing,
        mo_settings=mo_settings,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        th_ref=th_ref,
    )
