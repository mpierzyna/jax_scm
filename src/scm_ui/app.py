import bokeh.layouts as bl
import bokeh.models as bm
import bokeh.plotting as bp

from ic_bc_edit import MYNNEditor
from scm.grid import StaggeredGrid
from scm.interfaces import Simulation
from scm.io.local import out_to_ds
from scm.mo import MOSettings
from scm.mynn.model import init_model
from scm.time_stepping import simulate_adaptive_dt


class App:
    def __init__(self, H: float, Nz: int):
        self.grid = StaggeredGrid(H=H, Nz=Nz)
        self.editor = MYNNEditor(H=H, Nz=Nz)

        self.btn_run = bm.Button(label="Run!", button_type="primary")
        self.btn_run.on_click(self.run)

        self.tabs = bm.Tabs(
            tabs=[
                bm.TabPanel(child=self.editor.get_layout(), title="IC/BC editor"),
                bm.TabPanel(child=bl.column(self.btn_run), title="Model output"),
            ]
        )

    def run(self):
        # Run interpolation callback, so user sees which profiles are used
        for e in [*self.editor.ics.values(), *self.editor.bcs.values(), *self.editor.frc.values()]:
            e.interp_callback()

        ic = self.editor.get_ic()
        frc = self.editor.get_forcing()
        mo_settings = MOSettings(z0h=0.1, z0m=0.1)
        sim = Simulation(
            name="WebUI",
            init=ic,
            forcing=frc,
            grid=self.grid,
            mo_settings=mo_settings,
            t_start_s=0,
            t_end_s=24 * 60 * 60,  # todo: hardcode 24h
            th_ref=300.0,  # todo: expose in UI
        )

        model = init_model(sim)
        state_hist, diag_hist, mo_hist, t = simulate_adaptive_dt(
            model=model,
            sim=sim,
            dt_s_init=0.001,
            dt_s_max=1,
            cfl_max=0.05,
            dt_s_out=60 * 5,
            pbar=False,
        )
        ds = out_to_ds(state_hist, diag_hist, mo_hist, time=t / 60 / 60, grid=sim.grid)
        ds.to_netcdf(f"out_ui.nc")

    def get_layout(self):
        return self.tabs


app = App(H=2000, Nz=100)
bp.curdoc().add_root(app.get_layout())
