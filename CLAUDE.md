# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hill Rep Analysis is a Python CLI tool that automatically detects and analyzes staircase/hill repetitions from Garmin FIT files. It outputs per-rep statistics as JSON and an interactive HTML dashboard.

## Commands

```bash
# Install dependencies
pip install fitparse pandas numpy scipy

# Run analysis on a FIT file
python analyze.py <fit_file>
python analyze.py <fit_file> --output <output_dir>
```

## Architecture

The analysis pipeline in `analyze.py` follows these steps:

1. **FIT Parsing** (`parse_fit`): Extracts timestamps, GPS coordinates (converted from semicircles), altitude, heart rate, and distance
2. **Sawtooth Detection** (`detect_sawtooth_region`): Identifies the rep region using rolling altitude standard deviation (threshold: 3.0m over 100-point window)
3. **GPS Boundary Detection** (`find_gps_passes`): Finds first/last passes near the staircase bottom coordinates using peak detection on inverted distance
4. **Valley Detection** (`find_elevation_valleys`): Locates rep boundaries at altitude valleys using Savitzky-Golay smoothed data, then snaps to raw minima
5. **Statistics** (`compute_rep_stats`): Splits each rep at the altitude peak into uphill/downhill segments; calculates duration, distance, elevation gain/loss, HR, and pace

The HTML dashboard (`template.html`) receives data via JSON injection at the `REPLACE_DATA` placeholder. It renders summary cards, an altitude chart with valley markers, and tabbed tables for full/uphill/downhill rep details.

## Configuration

Staircase bottom GPS coordinates are hardcoded at the top of `analyze.py`:

```python
BOTTOM_LAT = 50.983862
BOTTOM_LON = 5.048131
```

Modify these values for different locations.
