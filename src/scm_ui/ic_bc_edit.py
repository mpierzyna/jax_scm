import abc
import enum
from typing import Literal, Tuple

import bokeh.plotting as bp
import jax.numpy as jnp
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, DataTable, PointDrawTool, RadioButtonGroup, TableColumn
from scipy.interpolate import CubicSpline

from scm.forcing.interp import get_ts_interp_fn
from scm.interfaces import Forcing
from scm.mynn.interfaces import ProgVarsMYNN


class InterpType(enum.StrEnum):
    LINEAR = "linear"
    CUBIC = "cubic"


class Editor(abc.ABC):
    """Generic editor for adding (x, y) points and interpolating between them.
    In subclasses, either x or y is set as the input variable, and the other is interpolated as a function of it.
    """

    def __init__(
        self,
        name: str,
        input_label: str,
        height: int,
        width: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        val_default: float,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the variable being edited (e.g., "u", "T_s").
        input_label : str
            Label for the input variable (e.g., "z, m" for profiles)
        height : int
            Height of the editor figure in pixels.
        width : int
            Width of the editor figure in pixels.
        x_range : Tuple[float, float]
            Range of x-axis for the figure.
        y_range : Tuple[float, float]
            Range of y-axis for the figure.
        val_default : float
            Default value to use for interpolation when no points are defined.
        x : np.ndarray | None
            x values to interpolate over if y is target
        y : np.ndarray | None
            y values to interpolate over if x is target
        """
        # Ensure that either x or y are set as interpolation is based on them
        assert (x is None) != (y is None), "Either x or y must be set, but not both or neither."
        self.x = x
        self.y = y
        self.val_default = val_default

        # Init data stores
        self.pts = ColumnDataSource({"x": [], "y": []})
        self.line = ColumnDataSource({"x": [], "y": []})
        self._interp_type: InterpType = InterpType.LINEAR

        # Setup figure
        p = bp.figure(height=height, width=width, tools="reset", title=name)
        p.x_range.start, p.x_range.end = x_range
        p.y_range.start, p.y_range.end = y_range
        p.on_event("reset", lambda _: self._reset())

        # Setup point and line renderers
        s = p.scatter(x="x", y="y", source=self.pts, size=10)
        p.line(x="x", y="y", source=self.line, line_width=2)

        # Set axis labels
        if self.x_is_input:
            p.xaxis.axis_label = input_label
            p.yaxis.axis_label = name
        else:
            p.xaxis.axis_label = name
            p.yaxis.axis_label = input_label

        # Add point draw tool
        tool = PointDrawTool(renderers=[s], empty_value=0)
        p.add_tools(tool)
        p.toolbar.active_tap = tool

        # Add datatable for detailed editing of point positions
        x_col = TableColumn(field="x", title=input_label if self.x_is_input else name)
        y_col = TableColumn(field="y", title=name if self.x_is_input else input_label)
        table = DataTable(
            source=self.pts,
            editable=True,
            columns=[x_col, y_col] if self.x_is_input else [y_col, x_col],
            height=min(100, height),  # 100 shows ~3 rows.
            width=200,
        )

        # Button to trigger interpolation
        btn_interp = Button(label="Interpolate", button_type="success")
        btn_interp.on_click(self.interp_callback)

        # Toggle interpolation modes
        radio_interp = RadioButtonGroup(labels=[InterpType.LINEAR, InterpType.CUBIC], active=0)
        radio_interp.on_change("active", self._switch_interp)

        # Store references for layout
        self.p = p
        self.btn_interp = btn_interp
        self.radio_interp = radio_interp
        self.table = table

    @property
    def x_is_input(self) -> bool:
        """True if x is input, false if y is input."""
        return self.x is not None

    def _reset(self):
        self.pts.data = {"x": [], "y": []}
        self.line.data = {"x": [], "y": []}

    def _switch_interp(self, attr, old, new):
        self._interp_type = InterpType(self.radio_interp.labels[new])

    def interpolate(self) -> np.ndarray:
        if self.x_is_input:
            # Interpolate y as function of x (forcing/time series where x=t and y=value)
            inp = self.x
            pts_inp = self.pts.data["x"]
            pts_val = self.pts.data["y"]
        else:
            # Interpolate x as function of y (profiles, where y=z and x=value)
            inp = self.y
            pts_inp = self.pts.data["y"]
            pts_val = self.pts.data["x"]

        if len(pts_inp) < 2:
            # Cannot interpolate with less than two points
            return np.ones_like(inp) * self.val_default

        dep_sorted = np.argsort(pts_inp)
        pts_inp = np.array(pts_inp, dtype=float)[dep_sorted]
        pts_val = np.array(pts_val, dtype=float)[dep_sorted]

        if self._interp_type == InterpType.LINEAR:
            pts_val_interp = np.interp(inp, pts_inp, pts_val)
        elif self._interp_type == InterpType.CUBIC:
            cs = CubicSpline(pts_inp, pts_val, bc_type="not-a-knot")
            pts_val_interp = cs(inp)
        else:
            raise ValueError(f"Unsupported interpolation type: {self._interp_type}")

        return pts_val_interp

    def interp_callback(self):
        pts_val_interp = self.interpolate()
        if self.x is None:
            self.line.data = {"x": pts_val_interp, "y": self.y}
        else:
            self.line.data = {"x": self.x, "y": pts_val_interp}

    @abc.abstractmethod
    def get_layout(self):
        """Implement how layout is organized."""


class ProfileEditor(Editor):
    """Editor for vertical profiles."""

    def __init__(self, name: str, val_range: Tuple[float, float], z: np.ndarray, height: int = 500, width: int = 300):
        """
        Parameters
        ----------
        name : str
            Name of the variable being edited (e.g., "u", "T_s").
        val_range : Tuple[float, float]
            Range of values for the variable being edited, used to set x-axis limits.
        z : np.ndarray
            1D array of vertical levels (m) to interpolate over.
        height : int
            Height of the editor figure in pixels.
        width : int
            Width of the editor figure in pixels.
        """
        super().__init__(
            name,
            x_range=val_range,
            y_range=(z.min(), z.max()),
            y=z,
            input_label="z, m",
            height=height,
            width=width,
            val_default=(val_range[0] + val_range[1]) / 2,
        )
        self.table.width = width - 150

    def get_layout(self):
        return column(
            self.p,
            row(
                self.table,
                column(
                    self.radio_interp,
                    self.btn_interp,
                ),
            ),
        )


class BCEditor(Editor):
    """Editor for time series of boundary conditions"""

    def __init__(self, name: str, val_range: Tuple[float, float], t: np.ndarray, height: int = 200, width: int = 750):
        """
        Parameters
        ----------
        name : str
            Name of the variable being edited (e.g., "T_s", "w_qv").
        val_range : Tuple[float, float]
            Range of values for the variable being edited, used to set y-axis limits.
        t : np.ndarray
            1D array of time points (hours) to interpolate over.
        height : int
            Height of the editor figure in pixels.
        width : int
            Width of the editor figure in pixels.
        """
        super().__init__(
            name,
            x=t,
            x_range=(t.min(), t.max()),
            y_range=val_range,
            input_label="t, h",
            height=height,
            width=width,
            val_default=(val_range[0] + val_range[1]) / 2,
        )

    def get_layout(self):
        return row(
            self.p,
            column(
                self.table,
                row(
                    self.radio_interp,
                    self.btn_interp,
                ),
            ),
        )


class MYNNEditor:
    def __init__(self, H: float, Nz: int):
        z = np.linspace(0, H, Nz)
        time_h = np.linspace(0, 24, 100)  # hours of simulation time
        self.ics = {
            "u": ProfileEditor("u, m/s", (-10, 10), z),
            "v": ProfileEditor("v, m/s", (-10, 10), z),
            "th": ProfileEditor("th, K", (250, 300), z),
            "qv": ProfileEditor("q_v, g/kg", (-0.1, 0.1), z),
            "qke": ProfileEditor("qke, (m/s)^2", (0, 0.2), z),
        }
        self.bcs = {
            "T_s": BCEditor("T_s, K", (273 - 20, 273 + 20), time_h),
            "w_qv": BCEditor("w_qv, (m g) / (kg s)", (-0.1, 0.1), time_h),
        }
        self.frc = {
            "u_geo": ProfileEditor("u_geo, m/s", (-10, 10), z),
            "v_geo": ProfileEditor("v_geo, m/s", (-10, 10), z),
        }

        self.z = z
        self.time_h = time_h

    def get_ic(self) -> ProgVarsMYNN:
        return ProgVarsMYNN(
            u=jnp.array(self.ics["u"].interpolate()),
            v=jnp.array(self.ics["v"].interpolate()),
            th=jnp.array(self.ics["th"].interpolate()),
            qv=jnp.array(self.ics["qv"].interpolate()),
            qke=jnp.array(self.ics["qke"].interpolate()),
        )

    def get_forcing(self) -> Forcing:
        def _get_frc_fn(name: str, mode: Literal["profile", "bc"]):
            if mode == "profile":
                data = jnp.array(self.frc[name].interpolate())
                return lambda t_s: data  # constant forcing in time
            elif mode == "bc":
                data = jnp.array(self.bcs[name].interpolate())
                return get_ts_interp_fn(time_s=jnp.array(self.time_h) * 3600, data=data)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        return Forcing(
            f_c=1e-4,
            u_geo=_get_frc_fn("u_geo", "profile"),
            v_geo=_get_frc_fn("v_geo", "profile"),
            th_s=_get_frc_fn("T_s", "bc"),
            w_qv_s=_get_frc_fn("w_qv", "bc"),
        )

    def get_layout(self):
        return column(
            row(*[ic.get_layout() for ic in self.ics.values()]),  # initial conditions
            row(
                *[frc.get_layout() for frc in self.frc.values()],  # forcing profiles
                column(*[bc.get_layout() for bc in self.bcs.values()]),  # surface BCs
            ),
        )


# mynn_ic_bc = MYNNEditor(H=1000, Nz=50)
# bp.curdoc().add_root(mynn_ic_bc.get_layout())
