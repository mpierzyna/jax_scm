from __future__ import annotations
import numpy as np
import xarray as xr
import bokeh.plotting as bp
import bokeh.models as bm
import bokeh.layouts as bl

WIDTH = 1200

ds = xr.open_dataset("../out.nc")
ds["m"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
ds["d"] = np.rad2deg(np.arctan2(-ds["u"], -ds["v"]))

H = ds["zh"].values[-1]  # total depth
t_start, t_end = ds["time"].values[[0, -1]]

vars_1d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 1] + ["time"]
store_1d = bm.ColumnDataSource(data={v: ds[v].values for v in vars_1d})

vars_2d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 2]


class TsPlot:
    def __init__(self, var: str, linked: TsPlot | None = None):
        # Create figure with shared x_range if linked, otherwise default to time range
        fig_kwargs = {}
        if linked:
            fig_kwargs["x_range"] = linked.p.x_range

        p = bp.figure(height=300, width=WIDTH, **fig_kwargs)
        if not linked:
            p.x_range.range_padding = p.y_range.range_padding = 0
            p.x_range.bounds = (t_start, t_end)

        # Plot line
        line = p.line(x="time", y=var, source=store_1d)

        # Selector
        def _sel_callback(attr, old, new):
            line.glyph.y = new

        select = bm.Select(title="Select variable", options=vars_1d[:-1], value=var)
        select.on_change("value", _sel_callback)

        # Log switch
        def _log_callback(attr, old, new):
            if new:
                p.y_scale = bm.LogScale()
            else:
                p.y_scale = bm.LinearScale()

        log_toggle = bm.Switch(label="Log scale")
        log_toggle.on_change("active", _log_callback)

        self.p = p
        self.select = select
        self.log_toggle = log_toggle

    def get_layout(self):
        return bl.column(
            bl.row(self.select, self.log_toggle),
            self.p,
        )


class FieldPlot:
    CMAP_SEQ = "Magma11"
    CMAP_DIV = "Sunset11"

    def __init__(self, var: str):
        p = bp.figure(height=500, width=WIDTH)
        color_mapper = bm.LinearColorMapper(palette=self.CMAP_SEQ, low=0, high=1)  # dummy values
        img = p.image(
            image=[],  # will be populated by `_update_data` call
            x=t_start,
            y=0,
            dw=t_end - t_start,
            dh=H,
            color_mapper=color_mapper,
            level="image",
        )
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.x_range.bounds = (t_start, t_end)
        p.y_range.bounds = (0, H)

        # Color bar
        cbar = bm.ColorBar(color_mapper=color_mapper)
        p.add_layout(cbar, "right")

        # Color bar ranges selector
        vrange = bm.RangeSlider(start=0, end=1, value=(0, 1), step=10, title="Color range")  # dummy values
        vrange.on_change("value", self._vrange_callback)

        # Variable selector
        select = bm.Select(title="Select variable", options=vars_2d, value=var)
        select.on_change("value", self._sel_callback)

        # Log scale toggle
        log_toggle = bm.Switch(label="Log scale")
        log_toggle.on_change("active", self._log_callback)

        # Divergent colormap toggle
        div_toggle = bm.Switch(label="Divergent colors")
        div_toggle.on_change("active", self._div_color_callback)

        # Store references for layout
        self.p = p
        self.select = select
        self.log_toggle = log_toggle
        self.vrange = vrange
        self.div_toggle = div_toggle

        # Store references for updating
        self._img = img
        self._color_mapper = color_mapper

        # Finally, set data through high-level function to ensure all components are updated correctly
        self.var = var
        self._update_data(var=var, use_log=False)

    def _update_vmin_vmax(self, vmin: float, vmax: float):
        """High level; update colormap and range selector"""
        # Update colormapper
        self._color_mapper.low = vmin
        self._color_mapper.high = vmax

        # Update vrange slider
        self.vrange.start = vmin
        self.vrange.end = vmax
        self.vrange.value = (vmin, vmax)
        self.vrange.step = (vmax - vmin) / 250

    def _update_data(self, var: str | None, use_log: bool):
        """High level; update data"""
        if var is None:
            var = self.var  # reuse currently selected variable
        else:
            self.var = var  # update currently selected variable

        # Load data
        data = ds[var].values.T
        data = np.log10(data) if use_log else data
        data = np.where(np.isfinite(data), data, np.nan)  # log can introduce -inf, so force to NaN

        # Update image
        self._img.data_source.data["image"] = [data]
        self._update_vmin_vmax(vmin=np.nanmin(data), vmax=np.nanmax(data))

    def _sel_callback(self, attr, old, new_var):
        self._update_data(var=new_var, use_log=self.log_toggle.active)
        if self.div_toggle.active:
            self._div_color_callback(None, None, True)

    def _vrange_callback(self, attr, old, new):
        """Set colorbar to range from range selector"""
        new_min, new_max = new
        # Update colormapper
        if self.div_toggle.active:
            old_min, old_max = old
            if old_min != new_min:
                # left changed, update right to keep symmetric
                new_max = new_min * -1
            if old_max != new_max:
                # right changed, update left to keep symmetric
                new_min = new_max * -1
            self.vrange.value = (new_min, new_max)  # update slider to reflect symmetric change

        self._color_mapper.low = new_min
        self._color_mapper.high = new_max

    def _log_callback(self, attr, old, use_log):
        """Keep current variable, but update log scale"""
        self.div_toggle.disabled = use_log  # log doesn't make sense for divergent data
        if self.div_toggle.active and use_log:
            self.div_toggle.active = False  # turn off divergent colors if log is toggled on
        self._update_data(var=None, use_log=use_log)

    def _div_color_callback(self, attr, old, use_div):
        """Switch to divergent colormap if toggled on, otherwise use sequential."""
        # Get min and max values
        vmin = ds[self.var].values.min()
        vmax = ds[self.var].values.max()

        if use_div:
            self.log_toggle.active = False  # log doesn't make sense for divergent data
            self._color_mapper.palette = self.CMAP_DIV

            vsym = max(abs(vmin), abs(vmax))  # symmetric range around zero
            self._update_vmin_vmax(vmin=-vsym, vmax=vsym)
        else:
            self._color_mapper.palette = self.CMAP_SEQ
            self._update_vmin_vmax(vmin=vmin, vmax=vmax)

    def get_layout(self):
        return bl.column(
            bl.row(
                self.select,
                self.vrange,
                bl.column(
                    self.log_toggle,
                    self.div_toggle,
                ),
            ),
            self.p,
        )


bp.curdoc().add_root(
    bl.column(
        FieldPlot("m").get_layout(),
        TsPlot("mo_u_st").get_layout(),
    )
)
