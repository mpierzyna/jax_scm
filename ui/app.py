import bokeh.plotting as bp
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, PointDrawTool, RadioButtonGroup, DataTable, TableColumn
import numpy as np
import enum

from scipy.interpolate import CubicSpline


class InterpType(enum.StrEnum):
    LINEAR = "linear"
    CUBIC = "cubic"


class ProfileEditor:

    def __init__(self, name: str, val_range: tuple[float, float], z: np.ndarray):
        # Init data stores
        self.z = z
        self.pts = ColumnDataSource({"val": [], "z": []})
        self.line = ColumnDataSource({"val": [], "z": []})
        self._interp_type: InterpType = InterpType.LINEAR

        # Setup figure
        p = bp.figure(height=500, width=300, tools="reset", title=name)
        p.x_range.start, p.x_range.end = val_range
        p.y_range.start, p.y_range.end = z[0], z[-1]
        p.on_event("reset", lambda _: self._reset())

        # Setup point and line renderers
        s = p.scatter(x="val", y="z", source=self.pts, size=10)
        p.line(x="val", y="z", source=self.line, line_width=2)

        # Add point draw tool
        tool = PointDrawTool(renderers=[s], empty_value=0)
        p.add_tools(tool)
        p.toolbar.active_tap = tool

        # Add datatable for detailed editing of point positions
        dt = DataTable(
            source=self.pts,
            editable=True,
            columns=[
                TableColumn(field="z", title="z, m"),
                TableColumn(field="val", title=name),
            ],
            height=100,
            width=300,
        )

        # Add submit button
        b = Button(label="Interpolate", button_type="success")
        b.on_click(self._interpolate)

        # Add interpolation toggle
        rb = RadioButtonGroup(labels=[InterpType.LINEAR, InterpType.CUBIC], active=0)
        rb.on_change("active", self._switch_interp)

        # Store references for layout
        self.p = p
        self.b = b
        self.rb = rb
        self.dt = dt

    def _reset(self):
        self.pts.data = {"val": [], "z": []}
        self.line.data = {"val": [], "z": []}

    def _switch_interp(self, attr, old, new):
        self._interp_type = InterpType(self.rb.labels[new])

    def _interpolate(self):
        pts_z = self.pts.data["z"]
        z_order = np.argsort(pts_z)
        pts_z = np.array(pts_z, dtype=float)[z_order]
        pts_val = self.pts.data["val"]
        pts_val = np.array(pts_val, dtype=float)[z_order]

        if self._interp_type == InterpType.LINEAR:
            self.line.data = {"val": np.interp(self.z, pts_z, pts_val), "z": self.z}
        elif self._interp_type == InterpType.CUBIC:
            cs = CubicSpline(pts_z, pts_val, bc_type="natural")
            self.line.data = {"val": cs(self.z), "z": self.z}

    def get_layout(self):
        return column(self.p, self.dt, row(self.rb, self.b))


H = 2000
z = np.linspace(0, H, 100)
u_edit = ProfileEditor("u", (-10, 10), z)
v_edit = ProfileEditor("v", (-10, 10), z)
th_edit = ProfileEditor("theta", (250, 10), z)
qv_edit = ProfileEditor("q_v", (-0.1, 0.1), z)

# add the plot and the button to the document
bp.curdoc().add_root(
    row(
        u_edit.get_layout(),
        v_edit.get_layout(),
        th_edit.get_layout(),
        qv_edit.get_layout(),
    ),
)
