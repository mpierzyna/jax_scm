#!/bin/bash
MY_DIR=$(PWD)
REF_DIR="../../../validation/wangara"

# Update wangara fixture by running a simulation and copying results
cd ${REF_DIR}
uv run run.py
cp wangara_day33.nc ${MY_DIR}/out_cn.nc
cp namelist_cn.yaml ${MY_DIR}
