from __future__ import annotations

import bokeh.layouts as bl
import bokeh.models as bm
import bokeh.plotting as bp
import numpy as np
import xarray as xr
from bokeh.events import MouseMove

WIDTH = 1200
PROFILE_WIDTH = 250
H_FIELD = 500
H_TS = 300


class DatasetWrapper:
    def __init__(self, ds: xr.Dataset):
        ds = ds.copy()
        ds["m"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
        ds["d"] = np.rad2deg(np.arctan2(-ds["u"], -ds["v"])) % 360  # meteorological [0, 360]

        self.ds = ds
        self.H = ds["zh"].values[-1]
        self.t_start, self.t_end = ds["time"].values[[0, -1]]

        # Bokeh event.x for datetime axes is ms since epoch (float64).
        # For plain float time axes it matches the data units directly.
        # Store a float64 array in the same units so _on_hover comparisons always work.
        t_vals = ds["time"].values
        if np.issubdtype(t_vals.dtype, np.datetime64):
            self.t_numeric = t_vals.astype("datetime64[ms]").astype(np.float64)
        else:
            self.t_numeric = t_vals.astype(np.float64)

        self.vars_1d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 1 and not str(v).startswith("_")]
        self.vars_2d = [str(v) for v in ds.data_vars if len(ds[v].shape) == 2]

    def get_ts_store(self) -> bm.ColumnDataSource:
        return bm.ColumnDataSource(data={v: self.ds[v].values for v in self.vars_1d + ["time"]})


class TsPlot:
    def __init__(self, dw: DatasetWrapper, var: str, x_range=None):
        p = bp.figure(height=H_TS, width=WIDTH, x_range=x_range)
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.x_range.bounds = (dw.t_start, dw.t_end)
        p.xaxis.axis_label = "time (h)"
        p.yaxis.axis_label = var

        line = p.line(x="time", y=var, source=dw.get_ts_store())

        time_marker = bm.Span(
            location=dw.t_start,
            dimension="height",
            line_color="tomato",
            line_dash="dashed",
            line_width=1.5,
        )
        p.add_layout(time_marker)

        def _sel_callback(attr, old, new):
            line.glyph.y = new
            p.yaxis.axis_label = new

        select = bm.Select(title="Select variable", options=dw.vars_1d, value=var)
        select.on_change("value", _sel_callback)

        self.p = p
        self.select = select
        self.time_marker = time_marker

    def get_layout(self):
        return bl.column(self.select, self.p)


class FieldPlot:
    CMAP_SEQ = "Magma11"
    CMAP_SEQ_FINE = "Magma256"
    CMAP_DIV = "RdBu11"

    def __init__(self, dw: DatasetWrapper, var: str):
        p = bp.figure(height=H_FIELD, width=WIDTH)
        color_mapper = bm.LinearColorMapper(palette=self.CMAP_SEQ, low=0, high=1)
        img = p.image(
            image=[],
            x=dw.t_start,
            y=0,
            dw=dw.t_end - dw.t_start,
            dh=dw.H,
            color_mapper=color_mapper,
            level="image",
        )
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.x_range.bounds = (dw.t_start, dw.t_end)
        p.y_range.bounds = (0, dw.H)
        p.xaxis.axis_label = "time (h)"
        p.yaxis.axis_label = "height (m)"

        cbar = bm.ColorBar(color_mapper=color_mapper)
        p.add_layout(cbar, "right")

        vrange = bm.RangeSlider(start=0, end=1, value=(0, 1), step=0.01, title="Color range")
        vrange.on_change("value", self._vrange_callback)

        select = bm.Select(title="Select variable", options=dw.vars_2d, value=var)
        select.on_change("value", self._sel_callback)

        log_toggle = bm.Switch(label="Log scale")
        log_toggle.on_change("active", self._log_callback)

        div_toggle = bm.Switch(label="Divergent colors")
        div_toggle.on_change("active", self._div_color_callback)

        fine_toggle = bm.Switch(label="Smooth colors")
        fine_toggle.on_change("active", self._fine_color_callback)

        # Profile panel: shares y_range with main figure
        profile_source = bm.ColumnDataSource({"x": [], "y": []})
        pp = bp.figure(height=H_FIELD, width=PROFILE_WIDTH, y_range=p.y_range)
        pp.line(x="x", y="y", source=profile_source, line_width=2, color="steelblue")
        pp.x_range.range_padding = 0.1
        pp.xaxis.axis_label = var

        # Vertical cursor line on the time-height plot
        profile_cursor = bm.Span(
            dimension="height",
            line_color="white",
            line_dash="dashed",
            line_width=1.5,
        )
        p.add_layout(profile_cursor)

        self.p = p
        self.p_profile = pp
        self.select = select
        self.log_toggle = log_toggle
        self.vrange = vrange
        self.div_toggle = div_toggle
        self.fine_toggle = fine_toggle
        self._img = img
        self._color_mapper = color_mapper
        self._profile_source = profile_source

        self._vrange_updating = False
        self._hover_t_idx = -1
        self._profile_cursor = profile_cursor
        self._time_marker: bm.Span | None = None  # wired by ResultViewer after construction

        self.var = var
        self.dw = dw
        self._update_data(var=var, use_log=False)

        p.on_event(MouseMove, self._on_hover)

    def _current_palette(self) -> str:
        if self.div_toggle.active:
            return self.CMAP_DIV
        return self.CMAP_SEQ_FINE if self.fine_toggle.active else self.CMAP_SEQ

    def _update_vmin_vmax(self, vmin: float, vmax: float):
        self._color_mapper.low = vmin
        self._color_mapper.high = vmax
        self.vrange.start = vmin
        self.vrange.end = vmax
        self.vrange.value = (vmin, vmax)
        self.vrange.step = max((vmax - vmin) / 250, 1e-10)

    def _update_profile(self, t_idx: int):
        var_dims = self.dw.ds[self.var].dims
        z_coord = "zh" if "zh" in var_dims else "z"
        z_vals = self.dw.ds[z_coord].values

        vals = self.dw.ds[self.var].values[t_idx, :]
        if self.log_toggle.active:
            vals = np.log10(np.where(vals > 0, vals, np.nan))
            vals = np.where(np.isfinite(vals), vals, np.nan)

        self._profile_source.data = {"x": vals.tolist(), "y": z_vals.tolist()}
        self.p_profile.xaxis.axis_label = self.var

    def _update_data(self, var: str | None, use_log: bool):
        if var is None:
            var = self.var
        else:
            self.var = var

        data = self.dw.ds[var].values.T
        data = np.log10(data) if use_log else data
        data = np.where(np.isfinite(data), data, np.nan)

        self._img.data_source.data["image"] = [data]
        self._update_vmin_vmax(vmin=np.nanmin(data), vmax=np.nanmax(data))
        self._update_profile(t_idx=0)
        self._hover_t_idx = 0

    def _on_hover(self, event):
        t = event.x
        t_numeric = self.dw.t_numeric  # float64, same units as event.x
        if t < t_numeric[0] or t > t_numeric[-1]:
            return

        # Both cursors track continuously (cheap location updates)
        self._profile_cursor.location = t
        if self._time_marker is not None:
            self._time_marker.location = t

        # Profile only updates when time index changes
        t_idx = int(np.argmin(np.abs(t_numeric - t)))
        if t_idx == self._hover_t_idx:
            return
        self._hover_t_idx = t_idx
        self._update_profile(t_idx)

    def _sel_callback(self, attr, old, new_var):
        self._update_data(var=new_var, use_log=self.log_toggle.active)
        if self.div_toggle.active:
            self._div_color_callback(None, None, True)

    def _vrange_callback(self, attr, old, new):
        if self._vrange_updating:
            return

        new_min, new_max = new
        if self.div_toggle.active:
            old_min, old_max = old
            if old_min != new_min:
                new_max = new_min * -1
            elif old_max != new_max:
                new_min = new_max * -1

            self._vrange_updating = True
            self.vrange.value = (new_min, new_max)
            self._vrange_updating = False

        self._color_mapper.low = new_min
        self._color_mapper.high = new_max

    def _log_callback(self, attr, old, use_log):
        self.div_toggle.disabled = use_log
        if self.div_toggle.active and use_log:
            self.div_toggle.active = False
        self._update_data(var=None, use_log=use_log)

    def _div_color_callback(self, attr, old, use_div):
        vmin = self.dw.ds[self.var].values.min()
        vmax = self.dw.ds[self.var].values.max()

        if use_div:
            self.log_toggle.active = False
            self.fine_toggle.disabled = True
            self._color_mapper.palette = self.CMAP_DIV
            vsym = max(abs(vmin), abs(vmax))
            self._update_vmin_vmax(vmin=-vsym, vmax=vsym)
        else:
            self.fine_toggle.disabled = False
            self._color_mapper.palette = self._current_palette()
            self._update_vmin_vmax(vmin=vmin, vmax=vmax)

    def _fine_color_callback(self, attr, old, use_fine):
        self._color_mapper.palette = self._current_palette()

    def get_controls(self):
        return bl.row(
            self.select,
            self.vrange,
            bl.column(self.log_toggle, self.div_toggle, self.fine_toggle),
        )

    def get_layout(self):
        """Self-contained layout for standalone use (no TsPlot)."""
        return bl.column(
            self.get_controls(),
            bl.row(self.p, self.p_profile),
        )


class ResultViewer:
    def __init__(self, ds: xr.Dataset):
        self.dw = DatasetWrapper(ds)

    def get_layout(self):
        field = FieldPlot(dw=self.dw, var="m")
        ts = TsPlot(dw=self.dw, var="mo_u_st", x_range=field.p.x_range)
        field._time_marker = ts.time_marker

        # Hovering in either plot drives the same callback (shared x_range → same data units)
        ts.p.on_event(MouseMove, field._on_hover)

        return bl.column(
            field.get_controls(),
            bl.row(
                bl.column(field.p, ts.get_layout()),
                field.p_profile,
            ),
        )
