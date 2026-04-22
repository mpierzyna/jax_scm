#!/bin/bash
MY_DIR=$(PWD)
REF_DIR="../../../validation/andren1994"

# Update andren1994 fixture by running a simulation and copying results
cd ${REF_DIR}
uv run run.py
cp andren1994.nc ${MY_DIR}/out_cn.nc
cp namelist_cn.yaml ${MY_DIR}
