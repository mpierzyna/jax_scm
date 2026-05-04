import jax
import jax.numpy as jnp
import pytest

from scm.mo import BusingerDyerSimFuncs, MOResult, MOSettings, init_mo_sfc

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


# Set up common test fixtures
@pytest.fixture
def bd_mo_w_th_s():
    """Businger-Dyer model with prescribed surface heat flux"""
    return init_mo_sfc(z0m=0.1, z0h=0.01, z=5, sim_funcs=BusingerDyerSimFuncs(), prescribe="w_th_s")


@pytest.fixture
def bd_mo_th_s():
    """Businger-Dyer model with prescribed surface temp"""
    return init_mo_sfc(z0m=0.1, z0h=0.01, z=5, sim_funcs=BusingerDyerSimFuncs(), prescribe="th_s")


@pytest.fixture(params=[False, True])
def use_jit(request):
    """Fixture to run tests twice -- with JIT enabled and disabled.
    Start with False for debugging purposes.
    """
    # Store the original JIT setting to restore later
    original = jax.config.jax_disable_jit

    # Set JIT state based on parameter (note: jax_disable_jit is inverted)
    jax.config.update("jax_disable_jit", not request.param)

    # Yield the current JIT enabled state to the test
    yield request.param

    # Restore original setting after test completes
    jax.config.update("jax_disable_jit", original)


@pytest.fixture
def mo_settings() -> MOSettings:
    """Fixture for MO settings serialization test."""
    sf = BusingerDyerSimFuncs(b=10, gamma=10)
    return MOSettings(z0m=0.1, z0h=0.01, sim_funcs=sf)


def test_serialize_settings(mo_settings):
    """Test that MOSettings can be serialized and deserialized without errors."""
    serialized = mo_settings.serialize()
    deserialized = MOSettings.deserialize(serialized)
    assert mo_settings.z0m == deserialized.z0m
    assert mo_settings.z0h == deserialized.z0h
    assert isinstance(deserialized.sim_funcs, BusingerDyerSimFuncs)
    assert deserialized.sim_funcs.b_h == mo_settings.sim_funcs.b_h
    assert deserialized.sim_funcs.gamma_h == mo_settings.sim_funcs.gamma_h


def test_bd(use_jit):
    """Test that BD functions can be called without errors."""
    phi_m_fn, phi_h_fn, psi_m_fn, psi_h_fn = BusingerDyerSimFuncs().get_all_fns()
    for zeta in jnp.array([-1.0, 0.0, 1.0]):
        phi_m_fn(zeta)
        phi_h_fn(zeta)
        psi_m_fn(zeta)
        psi_h_fn(zeta)


def test_neutral_conditions(bd_mo_w_th_s, use_jit):
    """Test near-neutral conditions where w_th_s ≈ 0"""
    res: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.0, w_qv_s=0.0, qv_0=0)

    # In neutral conditions, L should be very large
    assert jnp.abs(res.L) > 1000
    # Friction velocity should be positive
    assert res.u_st > 0
    # Heat flux should match prescribed value
    assert jnp.isclose(res.w_th, 0.0)
    # # dth_dz should be close to zero for neutral conditions
    # assert jnp.abs(res.dth_dz) < 0.01


def test_unstable_conditions(bd_mo_w_th_s, use_jit):
    """Test unstable conditions with positive heat flux"""
    res: MOResult = bd_mo_w_th_s(u_0=3.0, v_0=0.0, th_0=290.0, w_th_s=0.1, w_qv_s=0.0, qv_0=0)

    # For unstable conditions, L should be negative
    assert res.L < 0
    # Heat flux should match prescribed value
    assert jnp.isclose(res.w_th, 0.1)
    # # Negative temperature gradient in unstable conditions
    # assert res.dth_dz < 0


def test_stable_conditions(bd_mo_w_th_s, use_jit):
    """Test stable conditions with negative heat flux"""
    res: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=-0.01, w_qv_s=0.0, qv_0=0)

    # For stable conditions, L should be positive
    assert res.L > 0
    # Heat flux should match prescribed value
    assert jnp.isclose(res.w_th, -0.01)
    # # Positive temperature gradient in stable conditions
    # assert res.dth_dz > 0


def test_wind_direction(bd_mo_w_th_s, use_jit):
    """Test that the model handles different wind directions correctly"""
    # Wind from east
    res_east = bd_mo_w_th_s(u_0=3.0, v_0=0.0, th_0=290.0, w_th_s=0.01, w_qv_s=0.0, qv_0=0)
    # Wind from north
    res_north = bd_mo_w_th_s(u_0=0.0, v_0=3.0, th_0=290.0, w_th_s=0.01, w_qv_s=0.0, qv_0=0)
    # Wind from northeast (same magnitude)
    u_ne = v_ne = 3.0 / jnp.sqrt(2)
    res_ne = bd_mo_w_th_s(u_0=u_ne, v_0=v_ne, th_0=290.0, w_th_s=0.01, w_qv_s=0.0, qv_0=0)

    # Friction velocity should be similar for same wind speed magnitude
    assert jnp.isclose(res_east.u_st, res_north.u_st, rtol=1e-5)
    assert jnp.isclose(res_east.u_st, res_ne.u_st, rtol=1e-5)


def test_wind_magnitude(bd_mo_w_th_s, use_jit):
    """Test different wind magnitudes"""
    # Light wind
    res_light: MOResult = bd_mo_w_th_s(u_0=1.0, v_0=0.0, th_0=290.0, w_th_s=0.01, w_qv_s=0.0, qv_0=0)
    # Moderate wind
    res_mod: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.01, w_qv_s=0.0, qv_0=0)
    # Strong wind
    res_strong: MOResult = bd_mo_w_th_s(u_0=10.0, v_0=0.0, th_0=290.0, w_th_s=0.01, w_qv_s=0.0, qv_0=0)

    # Friction velocity should increase with wind speed
    assert res_light.u_st < res_mod.u_st < res_strong.u_st
    # L should increase with wind speed for same heat flux (unstable)
    assert jnp.abs(res_light.L) < jnp.abs(res_mod.L) < jnp.abs(res_strong.L)


def test_extreme_conditions(use_jit):
    """Test extreme conditions that might challenge the iteration scheme"""
    bd_mo_w_th_s = init_mo_sfc(z0m=0.1, z0h=0.01, z=5, sim_funcs=BusingerDyerSimFuncs(), prescribe="w_th_s")

    # Very strong instability (large positive heat flux, low wind)
    res_unst: MOResult = bd_mo_w_th_s(u_0=0.5, v_0=0.0, th_0=290.0, w_th_s=0.3, w_qv_s=0.0, qv_0=0)

    # Should converge to a reasonable L value
    assert not jnp.isnan(res_unst.L)
    assert res_unst.L < 0  # Unstable

    # Very strong stability (large negative heat flux)
    res_stab = bd_mo_w_th_s(u_0=0.5, v_0=0.0, th_0=290.0, w_th_s=-0.3, w_qv_s=0.0, qv_0=0)

    # Should converge to a reasonable L value
    assert not jnp.isnan(res_stab.L)
    assert res_stab.L > 0  # Stable


def test_10m_wind_and_2m_temp(bd_mo_w_th_s, use_jit):
    """Test the diagnostic 10m wind and 2m temperature outputs"""
    res: MOResult = bd_mo_w_th_s(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.01, w_qv_s=0.0, qv_0=0)

    # 10m wind should be positive but less than the reference height wind (due to log profile)
    assert 0 < res.m10 < 6.0

    # In unstable conditions with upward heat flux, 2m temperature should be between
    # surface temperature and reference height temperature
    assert min(res.th_s, 290.0) <= res.th2 <= max(res.th_s, 290.0)


def test_prescribe_th_s(bd_mo_th_s, use_jit):
    """Test the model with prescribed surface temperature"""
    res_stab: MOResult = bd_mo_th_s(u_0=5.0, v_0=0.0, th_0=295.0, th_s=290.0, w_qv_s=0.0, qv_0=0)
    res_neut: MOResult = bd_mo_th_s(u_0=5.0, v_0=0.0, th_0=290.0, th_s=290.0, w_qv_s=0.0, qv_0=0)
    res_unst: MOResult = bd_mo_th_s(u_0=5.0, v_0=0.0, th_0=290.0, th_s=295.0, w_qv_s=0.0, qv_0=0)

    assert jnp.isclose(res_neut.w_th, 0)
    assert res_unst.w_th > 0
    assert res_stab.w_th < 0


def test_sukanta_matlab(use_jit):
    """Compare with Sukanta's Matlab results"""
    z = 12.9032 / 2
    mo = init_mo_sfc(
        z0m=0.1,
        z0h=0.1,
        sim_funcs=BusingerDyerSimFuncs(b=5.0, gamma=15.0),
        z=z,
        prescribe="w_th_s",
    )

    res = mo(u_0=8, v_0=0, th_0=265, w_th_s=-0.08, w_qv_s=0.0, qv_0=0)
    assert jnp.isclose(res.u_st, 0.751988632621145)
    assert jnp.isclose(res.L, 3.589721161447895e02)
    # assert jnp.isclose(res.du_dz, 0.317580713832262)
    # assert jnp.isclose(res.dv_dz, 0.0)
    # assert jnp.isclose(res.dth_dz, 0.044928462436926)

    res: MOResult = mo(u_0=8, v_0=0, th_0=265, w_th_s=+0.08, w_qv_s=0.0, qv_0=0)
    assert jnp.isclose(res.u_st, 0.778360675518376)
    assert jnp.isclose(res.L, -3.980792561582825e02)
    # assert jnp.isclose(res.du_dz, 0.285643618090545)
    # assert jnp.isclose(res.dv_dz, 0.0)
    # assert jnp.isclose(res.dth_dz, -0.035721087489367)
    # assert jnp.isclose(res.m10, 8.852801010222086)  # sukanta doesn't reevaluate psi for 10m


def test_moisture_flux(use_jit):
    """Test that moisture fluxes are handled correctly"""
    bd_mo = init_mo_sfc(z0m=0.1, z0h=0.01, z=5, sim_funcs=BusingerDyerSimFuncs(), prescribe="w_th_s")

    res: MOResult = bd_mo(u_0=5.0, v_0=0.0, th_0=290.0, w_th_s=0.01, w_qv_s=0.005, qv_0=0.01)

    assert res.w_th < res.w_thv
