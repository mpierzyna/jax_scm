#!/bin/bash
CASES=(
    "andren1994"
    "gabls1"
    "wangara"
)

if [ -d "_reports_old" ]; then
    echo "Old reports found. Check if you still need them and then move or delete the _reports_old directory."
    exit 1
fi

# Backup old reports
mkdir _reports_old
for CASE in "${CASES[@]}"; do
    cp ${CASE}/report*.html _reports_old/
done

# Run all cases
for CASE in "${CASES[@]}"; do
    echo "Running case: $CASE"
    cd "$CASE" || exit
    uv run run.py &
    cd ..
done

# Wait for all background processes to finish
wait