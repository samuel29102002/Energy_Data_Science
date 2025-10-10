#!/bin/bash
# Convenience script to run the Dash app
# Usage: ./run.sh

cd "$(dirname "$0")"
python -m src.dash_app.app
