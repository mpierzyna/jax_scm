# Ensemble Example

Demonstrates running a parameter sweep over the MYNN B1 constant using `jax.vmap` to execute all ensemble members 
in parallel on a single device. This can be used for any type of ensemble simulation, e.g., using different initial 
conditions or model configurations.

## What it does

The script runs the GABLS1 case with 8 different values of the turbulence length scale parameter B1 (10–45, step 5) 
simultaneously. Results are saved to `ens_out.nc`. Three fields are plotted to present the ensemble diversity.

## Run

```bash
uv run python run_ensemble.py
```

## Key pattern

```python
def simulate_member(b1) -> Output:
    return simulate(model=model, sim=sim, cfg=cfg, params=MYNNParams(B1=b1))

B1_space = jnp.array([10, 15, 20, 25, 30, 35, 40, 45])
ens_out = jax.vmap(simulate_member)(B1_space)
```

`vmap` vectorizes `simulate_member` over the leading B1 axis, running all members in a single JIT-compiled call. 
`log_level` must be set to `LogLevel.SILENT` in the `Namelist` for `simulate` to compile under `vmap`.
