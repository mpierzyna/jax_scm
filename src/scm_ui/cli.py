import click
import xarray as xr
from bokeh.server.server import Server

from scm_ui.result_viewer import ResultViewer


def start_server(app_fn):
    # Setup server
    server = Server({"/": app_fn})
    server.start()
    click.echo(f"Server started at http://{server.address or 'localhost'}:{server.port}")

    # Start IO loop
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


@click.group()
def cli(): ...


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False))
def view(filepath: str):
    """View SCM results from a NetCDF file."""
    ds = xr.open_dataset(filepath)
    rv = ResultViewer(ds=ds)
    name = ds.attrs.get("name", "<no name>")

    def app(doc):
        doc.title = f"SCM: {filepath} ({name})"
        doc.add_root(rv.get_layout())

    start_server(app)
