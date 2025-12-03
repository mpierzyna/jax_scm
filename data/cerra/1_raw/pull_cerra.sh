#!/usr/bin/env bash
# Use rclone to copy CERRA files defined in `include.txt` from remote to current directory
rclone copy --include-from includes.txt --transfers=8 --progress tudelft:staff-umbrella/HBaki/CERRA .