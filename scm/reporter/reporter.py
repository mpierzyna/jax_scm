from __future__ import annotations

import base64
import datetime
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from .base import fig_to_png_base64, fig_to_svg_base64, env

# Load templates
tmpl_base = env.get_template("_base.html")
tmpl_table = env.get_template("_table.html")
tmpl_img = env.get_template("_img.html")
tmpl_li = env.get_template("_li.html")
tmpl_html_fig = env.get_template("_html_fig.html")


class BaseReport:
    def __init__(self, title: str, path: str | pathlib.Path, base_style_css: Dict = None):
        self.title = title
        self.path = path
        self.base_style_css = base_style_css or {}

        self.rendered_elements = []
        self.fig_counter = 0

        self.add_heading(title, level=1)

    def __enter__(self):
        """Context manager automatically saves report on exit (helpful if rendering crashes)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager automatically saves report on exit (helpful if rendering crashes)."""
        if exc_type is not None:
            print("Rendering failed. Saving error message to report.")
            self.add_text(f"Error: {exc_type} - {exc_val}")
            self.add_text(f"Traceback: {exc_tb}")
            self.save()

            raise exc_type(exc_val)
        else:
            self.save()

    def add_heading(self, heading: str, level: int) -> None:
        self.rendered_elements.append(f"<h{level}>{heading}</h{level}>")

    def add_text(self, text: str) -> None:
        self.rendered_elements.append(f"<p>{text}</p>")

    def add_hr(self) -> None:
        self.rendered_elements.append('<hr style="clear: both;"/>')

    def add_plotly_fig(self, fig: "go.Figure", caption: str = None, style: Dict = None) -> None:
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not available. Please install it to use this feature.")

        if caption is not None:
            self.fig_counter += 1

        fig_data = fig.to_html(full_html=False, include_plotlyjs="cdn")
        self.rendered_elements.append(
            tmpl_html_fig.render(
                fig=fig_data,
                class_name="plotly",
                counter=self.fig_counter,
                caption=caption,
                style=style or {},
            )
        )

    def add_mpl_fig(
        self,
        fig: plt.Figure,
        caption: str = None,
        fig_format: str = "png",
        style: Dict = None,
        autoclose: bool = True,
        **save_kwargs,
    ) -> plt.Figure | None:
        """Add matplotlib figure to report.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to add.
        caption : str, optional
            Caption for the figure.
        fig_format : str, optional
            Format to save the figure in. Options are "png" and "svg". Default is "png".
        style : Dict, optional
            CSS style to apply to the figure.
        autoclose : bool, optional
            Whether to automatically close the figure after adding it to the report. Default is True.
            Set this to False if you plan to reuse the figure, e.g., for saving to disk later.
        **save_kwargs
            Additional keyword arguments to pass to the figure saving function.

        Returns
        -------
        plt.Figure | None
            Returns the figure if autoclose is False, otherwise returns None.
        """
        match fig_format:
            case "png":
                fig_data = fig_to_png_base64(fig, **save_kwargs)
            case "svg":
                fig_data = fig_to_svg_base64(fig, **save_kwargs)
            case _:
                raise ValueError(f"Unknown figure format {fig_format}.")

        self.add_img(img_base64=fig_data, fig_format=fig_format, caption=caption, style=style, class_name="mpl")

        if autoclose:
            plt.close(fig)
            return None
        else:
            return fig

    def add_img(
        self,
        *,
        img_bytes: bytes = None,
        img_base64: str = None,
        fig_format: str,
        caption: str = None,
        style: Dict = None,
        class_name: str = None,
    ) -> None:
        """Add image to report. Can be provided as bytes or base64 string."""
        # Only count figures with captions
        if caption is not None:
            self.fig_counter += 1

        # If input is bytes, convert to base64
        if (img_bytes is None) and (img_base64 is None):
            raise ValueError("Either img_bytes or img_base64 must be provided.")
        if (img_bytes is not None) and (img_base64 is not None):
            raise ValueError("Only one of img_bytes or img_base64 can be provided.")
        if img_bytes is not None:
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        if fig_format == "svg":
            fig_format = "svg+xml"

        self.rendered_elements.append(
            tmpl_img.render(
                data=img_base64,
                format=fig_format,
                counter=self.fig_counter,
                caption=caption,
                style=style or {},
                class_name=class_name or "",
            )
        )

    def add_list(self, items: List[str]) -> None:
        """Add list to report (only single level for now)"""
        self.rendered_elements.append(tmpl_li.render(items=items))

    def add_dataframe(self, df: pd.DataFrame, caption: str = None) -> None:
        """Add dataframe to report."""
        self.rendered_elements.append(tmpl_table.render(table=df.to_html(), caption=caption))

    def save(self, path_overwrite: str | pathlib.Path = None) -> None:
        """Save report to file."""
        # Add rendering timestamp
        self.add_hr()
        self.add_text("Rendered on " + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))

        path = self.path if path_overwrite is None else path_overwrite
        with open(path, "w") as f:
            f.write(
                tmpl_base.render(
                    title=self.title,
                    body="\n".join(self.rendered_elements),
                    style=self.base_style_css,
                )
            )
