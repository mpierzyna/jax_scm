#!/bin/bash
MY_DIR="$(PWD)"
REF_DIR="../../../validation/gabls1"

# Update GABLS1 fixture by running a simulation and copying results
cd "${REF_DIR}"
uv run run.py
cp out*.nc namelist*.yaml "${MY_DIR}"
