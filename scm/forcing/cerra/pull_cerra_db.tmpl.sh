#!/usr/bin/env bash
# Use rclone to copy CERRA files defined in `include.txt` from remote to current directory
rclone copy --include-from includes.txt --transfers={{ n_transfers }} {{ progress }} {{ remote_path }} .