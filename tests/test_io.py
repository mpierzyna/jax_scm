from scm.config import Namelist, TimeIntMethod
from scm.examples.gabls1 import get_gabls1
from scm.io.local import out_to_ds
from scm.mynn.io import sim_from_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate

# Due to error accumulation over time, maximum errors quite high.
# Set variable dependent thresholds here.
MAX_ERR_THRESHOLDS = {
    "v_w": 5e-2,
    "w_th": 7e-2,
    "w_thv": 7e-2,
    "w_qke": 3e-2,
    "th_th": 4e-1,
    "L": 5,
    "L_S": 1e-2,
    "Km": 2e-2,
    "Kh": 3e-2,
    "Kq": 3e-2,
    "qke_P_B": 6e-2,
    "mo_zeta": 2e-2,
}


def test_sim_from_ds():
    # Test that a simulation can be restored from an output dataset, and that the new output matches the original.
    cfg = Namelist(time_int=TimeIntMethod.IMPLICIT, dt_s=1.0)

    # Run simulation normally
    sim = get_gabls1()
    model = init_model(sim, cfg=cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out=out, sim=sim)

    # Restore simulation from output dataset
    sim_ = sim_from_ds(ds)
    model_ = init_model(sim_, cfg=cfg)
    out_ = simulate(model=model_, sim=sim_, cfg=cfg)
    ds_ = out_to_ds(out=out_, sim=sim_)

    # Compare new output to original
    err = (ds - ds_) / ds.mean()
    for var in err.data_vars:
        # Skip forcing
        if "frc" in var:
            continue
        # Skip humidity because not in case
        if "qv" in var:
            continue
        if var in ["mo_zeta_err", "ct2"]:
            continue

        # Check that error is small
        assert err[var].max().item() < MAX_ERR_THRESHOLDS.get(str(var), 1e-2)
        assert err[var].mean().item() < 1e-3  # mean error below 0.1%
