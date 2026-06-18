#!/bin/bash
DIRS="$(find . -type d -not -path ".")"
for dir in $DIRS; do
  cd "$dir"
  ./update.sh
  cd ..
done
