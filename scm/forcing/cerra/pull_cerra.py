from __future__ import annotations

import datetime
import logging
import os
import pathlib
import re
from typing import List

import jinja2
import pandas as pd

FILE_ROOT = pathlib.Path(os.path.dirname(__file__))  # Root of this file

logger = logging.getLogger()


def render_template(template_path: str | pathlib.Path, **context) -> str:
    """Render a Jinja2 template with the given context."""
    template_path = pathlib.Path(template_path)
    template_str = template_path.read_text()
    template = jinja2.Template(template_str)
    return template.render(**context)


def load_cerra_filelist(fpath: pathlib.Path) -> pd.DataFrame:
    """Load CERRA file list from text file into pd.DataFrame.
    Obtained from remote server using `find /path/to/cerra -type f > cerra_filelist.txt`.
    """
    re_timestamp = re.compile(r"\d{4}_\d{2}_\d{2}-\d{2}")

    def _to_series(flist, name) -> pd.Series:
        """Convert list of CERRA filenames to pd.Series with timestamps as index."""
        # To speed up processing of big files, we flatten list first and use regex to find all timestamps at once
        flist_flat = ",".join(flist)
        timestamps = re_timestamp.findall(flist_flat)  #
        timestamps = pd.to_datetime(timestamps, format="%Y_%m_%d-%H")
        return pd.Series(flist, index=timestamps, name=name).sort_index()

    if fpath.suffix == ".gz":
        # gzipped text
        import gzip

        with gzip.open(fpath, "rt") as f:
            files = f.read().splitlines()
    elif fpath.suffix == ".txt":
        # raw text
        files = pathlib.Path(fpath).read_text().splitlines()
    else:
        raise ValueError("File must be .txt or .txt.gz")

    # There are four types: U10_V10, PRES, SFC, soil (from ERA5)
    cerra_pres = [f for f in files if f.endswith("PRES.grb")]
    cerra_uv = [f for f in files if f.endswith("U10_V10.grb")]
    cerra_sfc = [f for f in files if f.endswith("SFC.grb")]
    cerra_soil = [f for f in files if f.endswith("soil.grb")]

    df = pd.concat(
        [
            _to_series(cerra_pres, "PRES"),
            _to_series(cerra_uv, "UV"),
            _to_series(cerra_sfc, "SFC"),
            _to_series(cerra_soil, "SOIL"),
        ],
        axis=1,
    )

    # Ensure no missing files per timestamp
    if df.isnull().any().any():
        missing = df[df.isnull().any(axis=1)].isnull()
        logger.warning(f"Missing CERRA files for timestamps (False = missing):\n{~missing}")
        logger.warning("Records with missing files will be dropped.")
        df = df.dropna()

    return df


def setup(
    dates: List[datetime.date | slice | str],
    out_dir: pathlib.Path | str,
    remote_flist_path: pathlib.Path | str,
    remote_path: str,
    warmup_h: int = 6,
    n_transfers: int = 8,
):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    remote_flist_path = pathlib.Path(remote_flist_path)

    # Render script to pull data and save to stage dir
    rclone_sh = render_template(
        template_path=FILE_ROOT / "pull_cerra_db.tmpl.sh",
        progress="--progress",
        remote_path=remote_path,
        n_transfers=n_transfers,
    )
    (out_dir / "pull_cerra.sh").write_text(rclone_sh)
    logger.info(f"-> pull_cerra.sh rendered.")

    # Create include file for rclone pull
    df_full = load_cerra_filelist(remote_flist_path)
    df_inc = []
    for d in dates:
        if isinstance(d, datetime.date):
            d1 = datetime.datetime.combine(d, datetime.time(0, 0))
            d2 = d1 + datetime.timedelta(days=1)
            d1 -= datetime.timedelta(hours=warmup_h)
            d = slice(d1, d2)
        df_inc.append(df_full.loc[d])
    df_inc = pd.concat(df_inc)

    df_inc = df_inc.melt(value_name="path")
    (out_dir / "includes.txt").write_text("\n".join(df_inc["path"].to_list()))
    logger.info(f"-> rclone include file with {len(df_inc)} entries written to includes.txt.")
