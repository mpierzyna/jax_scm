import datetime
from scm.forcing.cerra import pull_cerra


target_dates = [
    datetime.date(2018, 3, 21),  # spring
    datetime.date(2018, 6, 21),  # summer
    datetime.date(2018, 9, 21),  # autumn
    datetime.date(2018, 12, 21),  # winter
]

if __name__ == "__main__":
    pull_cerra.setup(
        dates=target_dates,
        out_dir="1_raw",
        remote_flist_path="./CERRA_files.txt.gz",
        remote_path="tudelft:staff-umbrella/HBaki/CERRA",
    )
