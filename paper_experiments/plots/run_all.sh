#!/bin/bash
# Regenerate every paper figure. See README.md for environment setup.
set -e
cd "$(dirname "$0")"
for script in plot_*.py; do
    echo "== $script"
    python3 "$script"
done
