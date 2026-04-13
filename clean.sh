#!/bin/bash
# Remove all generated files except the best model (artifacts/best/)

rm -rf artifacts/models/
rm -rf artifacts/logs/
rm -rf artifacts/figures/
rm -rf runs/*

echo "Cleaned: artifacts/models, artifacts/logs, artifacts/figures, runs"
echo "Kept:    artifacts/best/"
