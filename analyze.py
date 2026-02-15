"""
Hill Rep Analysis — Detect and analyze staircase repetitions from a Garmin FIT file.

Usage:
    python analyze.py <fit_file> [--output <output_dir>]

Examples:
    python analyze.py workout.fit
    python analyze.py workout.fit --output results/

Requirements:
    pip install fitparse pandas numpy scipy

How it works:
    1. Parses the FIT file and extracts timestamps, GPS, altitude, HR, distance.
    2. Identifies the sawtooth (rep) region via rolling altitude standard deviation.
    3. Uses GPS proximity to the staircase bottom to find the first and last pass
       (defining the overall rep session boundaries).
    4. Within those boundaries, detects elevation valleys in the raw altitude data
       (smoothed only for robust detection, then snapped to actual raw minimum).
    5. Each interval between consecutive valleys = one rep (uphill + downhill).
    6. Exports per-rep statistics and an interactive HTML dashboard.
"""

import argparse
import json
import os
import sys

import fitparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ── Hill configurations ──
# Each hill has:
#   - bottom_lat/bottom_lon: GPS coordinates of the bottom (start/end of rep)
#   - top_lat/top_lon: GPS coordinates of the top (must pass by for valid rep)
HILLS = [
    {
        "id": "trappen-citadel-diest",
        "name": "Trappen Citadel Diest",
        "bottom_lat": 50.983862,
        "bottom_lon": 5.048131,
        "top_lat": 50.983525,
        "top_lon": 5.048036,
    },
    # Add more hills here in the future:
    # {
    #     "id": "another-hill",
    #     "name": "Another Hill Name",
    #     "bottom_lat": 51.123456,
    #     "bottom_lon": 4.567890,
    #     "top_lat": 51.123789,
    #     "top_lon": 4.567123,
    # },
]

# Default hill (for backward compatibility)
BOTTOM_LAT = HILLS[0]["bottom_lat"]
BOTTOM_LON = HILLS[0]["bottom_lon"]


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in metres."""
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def parse_timer_events(fitfile):
    """
    Extract timer start/stop events to identify paused periods.

    Returns a list of (pause_start, pause_end) datetime tuples.
    """
    events = []
    for record in fitfile.get_messages("event"):
        fields = {f.name: f.value for f in record.fields}
        if fields.get("event") == "timer" and fields.get("timer_trigger") == "manual":
            events.append({
                "timestamp": fields["timestamp"],
                "type": fields["event_type"],
            })

    # Find pause periods (stop_all -> start)
    pauses = []
    pause_start = None
    for event in events:
        if event["type"] == "stop_all":
            pause_start = event["timestamp"]
        elif event["type"] == "start" and pause_start is not None:
            pauses.append((pause_start, event["timestamp"]))
            pause_start = None

    return pauses


def calculate_elapsed_excluding_pauses(timestamps, pauses):
    """
    Calculate elapsed seconds excluding paused periods.

    For each timestamp, compute how much time has passed since the start,
    minus any pause duration that occurred before that timestamp.
    """
    if len(timestamps) == 0:
        return []

    start_time = timestamps.iloc[0]
    elapsed = []

    for ts in timestamps:
        # Total time since start
        total = (ts - start_time).total_seconds()

        # Subtract pause durations that occurred before this timestamp
        for pause_start, pause_end in pauses:
            if pause_end <= ts:
                # Full pause occurred before this point
                total -= (pause_end - pause_start).total_seconds()
            elif pause_start < ts < pause_end:
                # We're currently in a pause (shouldn't happen with clean data)
                total -= (ts - pause_start).total_seconds()

        elapsed.append(total)

    return elapsed


def parse_fit(path):
    """Read a .fit file and return a DataFrame with the fields we need."""
    fitfile = fitparse.FitFile(path)

    # First, get pause periods
    pauses = parse_timer_events(fitfile)

    # Re-parse to get records (generator was consumed)
    fitfile = fitparse.FitFile(path)
    rows = []
    for record in fitfile.get_messages("record"):
        row = {}
        for field in record.fields:
            row[field.name] = field.value
        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Calculate elapsed time excluding pauses
    df["elapsed_seconds"] = calculate_elapsed_excluding_pauses(df["timestamp"], pauses)

    df["lat"] = df["position_lat"] * (180 / 2**31)
    df["lon"] = df["position_long"] * (180 / 2**31)
    return df


def detect_sawtooth_region(alt_smooth):
    """Return (start_idx, end_idx) of the high-variance sawtooth region."""
    rolling_std = pd.Series(alt_smooth).rolling(100, center=True).std().fillna(0)
    mask = rolling_std > 3.0
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError("No sawtooth region detected — altitude variance is too low.")
    return int(indices[0]), int(indices[-1])


def find_gps_passes(df, sawtooth_start, sawtooth_end, max_dist_m=5):
    """Find indices where the runner passes closest to the staircase bottom."""
    dist = df["dist_to_bottom"].values[sawtooth_start : sawtooth_end + 1]
    passes_local, _ = find_peaks(-dist, distance=15, prominence=5)
    passes_global = passes_local + sawtooth_start
    close = passes_global[df["dist_to_bottom"].iloc[passes_global].values < max_dist_m]
    if len(close) < 2:
        raise ValueError(f"Only {len(close)} GPS pass(es) found near staircase bottom — need at least 2.")
    return int(close[0]), int(close[-1])


def find_elevation_valleys(alt, gps_first, gps_last, snap_window=8):
    """Detect raw-altitude valleys between the GPS-bounded region."""
    region = alt[gps_first : gps_last + 1]
    region_smooth = savgol_filter(region, window_length=11, polyorder=2)

    valleys_local, _ = find_peaks(-region_smooth, distance=20, prominence=5)
    valleys_global = valleys_local + gps_first

    # Snap each detected valley to the true raw-altitude minimum nearby
    snapped = []
    for v in valleys_global:
        ws = max(gps_first, v - snap_window)
        we = min(gps_last, v + snap_window)
        snapped.append(int(ws + np.argmin(alt[ws : we + 1])))

    # First boundary: raw min near first GPS pass
    ws, we = max(0, gps_first - 3), gps_first + 5
    first = int(ws + np.argmin(alt[ws : we + 1]))

    # Last boundary: raw min near last GPS pass (allow forward look — runner may still descend)
    ws, we = gps_last - 3, min(len(alt) - 1, gps_last + 8)
    last = int(ws + np.argmin(alt[ws : we + 1]))

    # Merge, sort, de-duplicate
    all_v = sorted(set([first] + snapped + [last]))
    all_v = [v for v in all_v if first <= v <= last]

    cleaned = [all_v[0]]
    for v in all_v[1:]:
        if v - cleaned[-1] < 10:
            if alt[v] < alt[cleaned[-1]]:
                cleaned[-1] = v
        else:
            cleaned.append(v)

    return cleaned


def fmt_dur(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def fmt_pace(dist_m, dur_s):
    if dist_m > 0 and dur_s > 0:
        pace_s = 1000 / (dist_m / dur_s)
        return f"{int(pace_s) // 60}:{int(pace_s) % 60:02d}"
    return "N/A"


def compute_rep_stats(df, alt, valleys):
    """Compute per-rep statistics given valley boundary indices.

    Uses elapsed_seconds which already has pause time excluded.
    """
    reps = []
    for i in range(len(valleys) - 1):
        si, ei = valleys[i], valleys[i + 1]
        seg_alt = alt[si : ei + 1]
        pi = si + int(np.argmax(seg_alt))

        # Uphill (use elapsed_seconds which excludes pauses)
        up = df.iloc[si : pi + 1]
        ua_s, ua_e = float(alt[si]), float(alt[pi])
        u_dur = float(up["elapsed_seconds"].iloc[-1] - up["elapsed_seconds"].iloc[0])
        u_dist = float(up["distance"].iloc[-1] - up["distance"].iloc[0])
        u_hr = float(up["heart_rate"].mean())

        # Downhill
        dn = df.iloc[pi : ei + 1]
        da_s, da_e = float(alt[pi]), float(alt[ei])
        d_dur = float(dn["elapsed_seconds"].iloc[-1] - dn["elapsed_seconds"].iloc[0])
        d_dist = float(dn["distance"].iloc[-1] - dn["distance"].iloc[0])
        d_hr = float(dn["heart_rate"].mean())

        # Full rep
        seg = df.iloc[si : ei + 1]
        f_dur = float(seg["elapsed_seconds"].iloc[-1] - seg["elapsed_seconds"].iloc[0])
        f_dist = float(seg["distance"].iloc[-1] - seg["distance"].iloc[0])
        f_hr = float(seg["heart_rate"].mean())

        reps.append({
            "rep": i + 1,
            "up_alt_start": round(ua_s, 1), "up_alt_end": round(ua_e, 1),
            "up_elev": round(ua_e - ua_s, 1),
            "up_dur_s": round(u_dur), "up_dur": fmt_dur(u_dur),
            "up_dist": round(u_dist, 1), "up_hr": int(round(u_hr)),
            "up_pace": fmt_pace(u_dist, u_dur),
            "dn_alt_start": round(da_s, 1), "dn_alt_end": round(da_e, 1),
            "dn_elev": round(da_s - da_e, 1),
            "dn_dur_s": round(d_dur), "dn_dur": fmt_dur(d_dur),
            "dn_dist": round(d_dist, 1), "dn_hr": int(round(d_hr)),
            "dn_pace": fmt_pace(d_dist, d_dur),
            "full_dur_s": round(f_dur), "full_dur": fmt_dur(f_dur),
            "full_dist": round(f_dist, 1), "full_hr": int(round(f_hr)),
            "si": int(si), "pi": int(pi), "ei": int(ei),
        })
    return reps


def build_chart_data(df, alt, valleys, sawtooth_start, sawtooth_end):
    """Sample the altitude profile and mark valleys for the HTML chart."""
    step = max(1, len(alt) // 600)
    chart = [
        {"t": round(float(df["elapsed_seconds"].iloc[i]), 1), "a": round(float(alt[i]), 1)}
        for i in range(0, len(alt), step)
    ]
    valley_markers = [
        {"t": round(float(df["elapsed_seconds"].iloc[v]), 1), "a": round(float(alt[v]), 1)}
        for v in valleys
    ]
    return chart, valley_markers


def generate_html(template_path, data, output_path):
    """Inject JSON data into the HTML template and write the output."""
    with open(template_path) as f:
        html = f.read()
    html = html.replace("REPLACE_DATA", json.dumps(data))
    html = html.replace("REPLACE_DATE", data["workout_date"])
    with open(output_path, "w") as f:
        f.write(html)


def detect_hill(df, saw_start, saw_end):
    """
    Detect which hill was used in this workout by finding which hill's
    bottom coordinates have the closest GPS passes.

    Returns the hill dict.
    """
    best_hill = None
    best_passes = 0

    for hill in HILLS:
        df["dist_to_bottom"] = haversine_m(
            df["lat"].values, df["lon"].values, hill["bottom_lat"], hill["bottom_lon"]
        )
        try:
            find_gps_passes(df, saw_start, saw_end, max_dist_m=10)
            # Count passes near the bottom
            dist = df["dist_to_bottom"].values[saw_start:saw_end + 1]
            passes_local, _ = find_peaks(-dist, distance=15, prominence=5)
            passes_global = passes_local + saw_start
            close_passes = len([p for p in passes_global if df["dist_to_bottom"].iloc[p] < 10])

            if close_passes > best_passes:
                best_passes = close_passes
                best_hill = hill
        except ValueError:
            continue

    if best_hill is None:
        # Fall back to first hill if none detected
        best_hill = HILLS[0]

    return best_hill


def validate_rep_reaches_top(df, si, ei, hill, max_dist_m=15):
    """
    Check if a rep segment actually passes near the top of the hill.

    A valid rep must get within max_dist_m meters of the top coordinates
    at some point between the start index (si) and end index (ei).

    Returns True if the rep passes by the top, False otherwise.
    """
    segment = df.iloc[si:ei + 1]
    dist_to_top = haversine_m(
        segment["lat"].values,
        segment["lon"].values,
        hill["top_lat"],
        hill["top_lon"]
    )
    # Rep is valid if it gets within max_dist_m of the top at any point
    return dist_to_top.min() < max_dist_m


def analyze_fit(path, hill=None):
    """
    Run full analysis on a FIT file and return the data dict.

    This is the main entry point for programmatic use (e.g., from a web server).
    Returns the same data structure that would be written to JSON by the CLI.

    Always returns valid data - if hill rep detection fails, returns workout
    stats with 0 reps instead of raising an exception.

    Args:
        path: Path to the FIT file
        hill: Optional hill dict to use. If None, uses default hill.
    """
    df = parse_fit(path)
    alt = df["enhanced_altitude"].values
    workout_date = df["timestamp"].iloc[0].strftime("%d %b %Y")

    # Compute workout-level stats (always available)
    workout_duration = float(df["elapsed_seconds"].iloc[-1])
    workout_distance = float(df["distance"].iloc[-1] - df["distance"].iloc[0])
    alt_diff = np.diff(alt)
    workout_total_ascent = float(np.sum(alt_diff[alt_diff > 0]))

    # Default hill if not specified
    if hill is None:
        hill = HILLS[0]

    # Try to detect hill reps - if any step fails, return workout with 0 reps
    reps = []
    chart = []
    valley_markers = []
    sawtooth_start_t = 0
    sawtooth_end_t = 0

    try:
        alt_smooth = savgol_filter(alt, window_length=15, polyorder=2)
        saw_start, saw_end = detect_sawtooth_region(alt_smooth)

        # Try to detect which hill based on GPS passes
        try:
            detected_hill = detect_hill(df, saw_start, saw_end)
            hill = detected_hill
        except ValueError:
            pass  # Keep default hill

        # Use the hill's bottom coordinates
        df["dist_to_bottom"] = haversine_m(
            df["lat"].values, df["lon"].values, hill["bottom_lat"], hill["bottom_lon"]
        )

        gps_first, gps_last = find_gps_passes(df, saw_start, saw_end)
        valleys = find_elevation_valleys(alt, gps_first, gps_last)

        # Compute stats for all detected rep segments
        all_reps = compute_rep_stats(df, alt, valleys)

        # Filter out reps that don't actually reach the top of the hill
        for rep in all_reps:
            if validate_rep_reaches_top(df, rep["si"], rep["ei"], hill):
                reps.append(rep)

        # Re-number the valid reps
        for i, rep in enumerate(reps):
            rep["rep"] = i + 1

        chart, valley_markers = build_chart_data(df, alt, valleys, saw_start, saw_end)
        sawtooth_start_t = round(float(df["elapsed_seconds"].iloc[saw_start]), 1)
        sawtooth_end_t = round(float(df["elapsed_seconds"].iloc[saw_end]), 1)

    except (ValueError, IndexError):
        # Hill rep detection failed - build basic chart without valleys
        step = max(1, len(alt) // 600)
        chart = [
            {"t": round(float(df["elapsed_seconds"].iloc[i]), 1), "a": round(float(alt[i]), 1)}
            for i in range(0, len(alt), step)
        ]

    # Compute rep summary stats (only for valid reps)
    rep_total_elevation = sum(r["up_elev"] for r in reps)
    avg_elevation = rep_total_elevation / len(reps) if reps else None
    avg_rep_time = sum(r["full_dur_s"] for r in reps) / len(reps) if reps else None

    return {
        "reps": reps,
        "chart": chart,
        "valleys": valley_markers,
        "sawtooth_start_t": sawtooth_start_t,
        "sawtooth_end_t": sawtooth_end_t,
        "workout_date": workout_date,
        "hill": {
            "id": hill["id"],
            "name": hill["name"],
        },
        "summary": {
            # Rep stats
            "num_reps": len(reps),
            "rep_total_elevation": round(rep_total_elevation, 1),
            "avg_elevation": round(avg_elevation, 1) if avg_elevation is not None else None,
            "avg_rep_time": round(avg_rep_time, 0) if avg_rep_time is not None else None,
            # Workout stats
            "workout_duration": round(workout_duration, 0),
            "workout_distance": round(workout_distance, 0),
            "workout_total_ascent": round(workout_total_ascent, 1),
        },
    }


def get_workout_summary(path):
    """
    Extract quick metadata from a FIT file without running full analysis.

    Returns:
        dict with filename, date, date_display
    """
    fitfile = fitparse.FitFile(path)
    timestamp = None
    for record in fitfile.get_messages("record"):
        for field in record.fields:
            if field.name == "timestamp":
                timestamp = field.value
                break
        if timestamp:
            break

    if timestamp:
        date_obj = pd.to_datetime(timestamp)
        return {
            "filename": os.path.basename(path),
            "date": date_obj.strftime("%Y-%m-%d"),
            "date_display": date_obj.strftime("%d %b %Y"),
        }
    return {
        "filename": os.path.basename(path),
        "date": "Unknown",
        "date_display": "Unknown",
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze hill reps from a Garmin FIT file.")
    parser.add_argument("fit_file", help="Path to the .fit file")
    parser.add_argument("--output", "-o", default=".", help="Output directory (default: current dir)")
    args = parser.parse_args()

    if not os.path.isfile(args.fit_file):
        print(f"Error: file not found: {args.fit_file}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.fit_file))[0]

    # ── Parse ──
    print(f"Parsing {args.fit_file} ...")
    df = parse_fit(args.fit_file)
    alt = df["enhanced_altitude"].values
    df["dist_to_bottom"] = haversine_m(df["lat"].values, df["lon"].values, BOTTOM_LAT, BOTTOM_LON)

    print(f"  {len(df)} data points, {df['elapsed_seconds'].iloc[-1]:.0f}s duration")
    print(f"  Altitude range: {alt.min():.1f} – {alt.max():.1f} m")

    # ── Detect ──
    alt_smooth = savgol_filter(alt, window_length=15, polyorder=2)
    saw_start, saw_end = detect_sawtooth_region(alt_smooth)
    print(f"  Sawtooth region: {df['elapsed_seconds'].iloc[saw_start]:.0f}s – {df['elapsed_seconds'].iloc[saw_end]:.0f}s")

    gps_first, gps_last = find_gps_passes(df, saw_start, saw_end)
    print(f"  GPS boundaries: idx {gps_first} – {gps_last}")

    valleys = find_elevation_valleys(alt, gps_first, gps_last)
    n_reps = len(valleys) - 1
    print(f"  Valleys: {len(valleys)}  →  Reps detected: {n_reps}")

    # ── Stats ──
    reps = compute_rep_stats(df, alt, valleys)

    avg_up = np.mean([r["up_elev"] for r in reps])
    avg_dn = np.mean([r["dn_elev"] for r in reps])
    avg_dur = np.mean([r["full_dur_s"] for r in reps])
    print(f"\n  Avg ↑ elevation: {avg_up:.1f} m")
    print(f"  Avg ↓ elevation: {avg_dn:.1f} m")
    print(f"  Avg rep duration: {avg_dur:.0f} s")

    # ── Export JSON ──
    chart, valley_markers = build_chart_data(df, alt, valleys, saw_start, saw_end)
    workout_date = df["timestamp"].iloc[0].strftime("%d %b %Y")

    data = {
        "reps": reps,
        "chart": chart,
        "valleys": valley_markers,
        "sawtooth_start_t": round(float(df["elapsed_seconds"].iloc[saw_start]), 1),
        "sawtooth_end_t": round(float(df["elapsed_seconds"].iloc[saw_end]), 1),
        "workout_date": workout_date,
    }

    json_path = os.path.join(args.output, f"{base}_reps.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  JSON saved: {json_path}")

    # ── Export HTML ──
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")
    if os.path.isfile(template_path):
        html_path = os.path.join(args.output, f"{base}_reps.html")
        generate_html(template_path, data, html_path)
        print(f"  HTML saved: {html_path}")
    else:
        print(f"  HTML template not found at {template_path} — skipping HTML export.")

    print(f"\nDone — {n_reps} reps analysed.")


if __name__ == "__main__":
    main()
