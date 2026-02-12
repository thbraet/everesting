# Hill Rep Analysis

Automatically detect and analyze staircase/hill repetitions from Garmin FIT files.

## How it works

1. Parses the FIT file for timestamps, GPS, altitude, heart rate, and distance
2. Identifies the sawtooth (rep) region via rolling altitude standard deviation
3. Uses GPS proximity to the staircase bottom to find the session boundaries (first arrival, last departure)
4. Detects elevation valleys in the raw altitude data for precise rep-to-rep transitions
5. Splits each rep into uphill and downhill segments
6. Exports per-rep statistics as JSON and an interactive HTML dashboard

## Setup

```bash
pip install fitparse pandas numpy scipy
```

## Usage

```bash
python analyze.py <fit_file> [--output <output_dir>]
```

**Examples:**
```bash
# Analyze a workout, output to current directory
python analyze.py 21820214233_ACTIVITY.fit

# Analyze and save results to a specific folder
python analyze.py 20789832636_ACTIVITY.fit --output results/
```

**Output files:**
- `<filename>_reps.json` — raw data for all detected reps
- `<filename>_reps.html` — interactive dashboard (open in any browser)

## Configuration

The staircase bottom GPS coordinates are defined at the top of `analyze.py`:

```python
BOTTOM_LAT = 50.983862
BOTTOM_LON = 5.048131
```

Update these if you use a different staircase location.

## Files

| File            | Description                                      |
|-----------------|--------------------------------------------------|
| `analyze.py`    | Main analysis script (CLI)                       |
| `template.html` | HTML dashboard template (used by analyze.py)     |
