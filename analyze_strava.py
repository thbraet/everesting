#!/usr/bin/env python3
"""
Hill Rep Analysis — Analyze staircase repetitions from Strava activity data.

This version uses Strava's official elevation data (with their smoothing applied),
which is what the Everesting challenge validates against.

Usage:
    python analyze_strava.py                    # Analyze all activities
    python analyze_strava.py --activity <id>   # Analyze specific activity
    python analyze_strava.py --output <dir>    # Specify output directory

Requirements:
    pip install pandas numpy scipy
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ── Hill configurations ──
HILLS = [
    {
        "id": "trappen-citadel-diest",
        "name": "Trappen Citadel Diest",
        "bottom_lat": 50.983862,
        "bottom_lon": 5.048131,
        "top_lat": 50.983525,
        "top_lon": 5.048036,
    },
]

ACTIVITIES_FILE = os.path.join(os.path.dirname(__file__), "strava_activities.json")


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in metres."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.atleast_1d, [lat1, lon1, lat2, lon2])
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def strava_to_dataframe(activity):
    """Convert Strava activity with streams to a DataFrame."""
    streams = activity.get("streams")
    if not streams:
        return None

    # Extract latlng into separate lat/lon columns
    latlng = streams.get("latlng", [])
    if not latlng:
        return None

    df = pd.DataFrame({
        "time": streams.get("time", []),
        "distance": streams.get("distance", []),
        "altitude": streams.get("altitude", []),
        "heart_rate": streams.get("heartrate", [0] * len(latlng)),
        "lat": [ll[0] for ll in latlng],
        "lon": [ll[1] for ll in latlng],
    })

    # Fill missing heart rate with 0
    if len(df["heart_rate"]) == 0:
        df["heart_rate"] = [0] * len(df)

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

    # Last boundary: raw min near last GPS pass
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
    """Compute per-rep statistics given valley boundary indices."""
    reps = []
    for i in range(len(valleys) - 1):
        si, ei = valleys[i], valleys[i + 1]
        seg_alt = alt[si : ei + 1]
        pi = si + int(np.argmax(seg_alt))

        # Uphill
        up = df.iloc[si : pi + 1]
        ua_s, ua_e = float(alt[si]), float(alt[pi])
        u_dur = float(up["time"].iloc[-1] - up["time"].iloc[0])
        u_dist = float(up["distance"].iloc[-1] - up["distance"].iloc[0])
        u_hr = float(up["heart_rate"].mean()) if up["heart_rate"].sum() > 0 else 0

        # Downhill
        dn = df.iloc[pi : ei + 1]
        da_s, da_e = float(alt[pi]), float(alt[ei])
        d_dur = float(dn["time"].iloc[-1] - dn["time"].iloc[0])
        d_dist = float(dn["distance"].iloc[-1] - dn["distance"].iloc[0])
        d_hr = float(dn["heart_rate"].mean()) if dn["heart_rate"].sum() > 0 else 0

        # Full rep
        seg = df.iloc[si : ei + 1]
        f_dur = float(seg["time"].iloc[-1] - seg["time"].iloc[0])
        f_dist = float(seg["distance"].iloc[-1] - seg["distance"].iloc[0])
        f_hr = float(seg["heart_rate"].mean()) if seg["heart_rate"].sum() > 0 else 0

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


def validate_rep_reaches_top(df, si, ei, hill, max_dist_m=15):
    """Check if a rep segment actually passes near the top of the hill."""
    segment = df.iloc[si:ei + 1]
    dist_to_top = haversine_m(
        segment["lat"].values,
        segment["lon"].values,
        hill["top_lat"],
        hill["top_lon"]
    )
    return dist_to_top.min() < max_dist_m


def detect_hill(df, saw_start, saw_end):
    """Detect which hill was used in this workout."""
    best_hill = None
    best_passes = 0

    for hill in HILLS:
        df["dist_to_bottom"] = haversine_m(
            df["lat"].values, df["lon"].values, hill["bottom_lat"], hill["bottom_lon"]
        )
        try:
            find_gps_passes(df, saw_start, saw_end, max_dist_m=10)
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
        best_hill = HILLS[0]

    return best_hill


def analyze_strava_activity(activity, hill=None):
    """
    Run full analysis on a Strava activity and return the data dict.

    Uses Strava's official elevation data which has their smoothing applied.
    This is what the Everesting challenge validates against.
    """
    df = strava_to_dataframe(activity)
    if df is None or len(df) < 100:
        return None

    alt = df["altitude"].values

    # Parse workout date
    start_date = activity.get("start_date_local", "")
    try:
        date_obj = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        workout_date = date_obj.strftime("%d %b %Y")
    except (ValueError, AttributeError):
        workout_date = "Unknown"

    # Use Strava's official elevation gain (this is what Everesting uses!)
    strava_elevation_gain = activity.get("total_elevation_gain", 0)

    # Workout-level stats from Strava
    workout_duration = activity.get("elapsed_time", 0)
    workout_distance = activity.get("distance", 0)

    # Default hill if not specified
    if hill is None:
        hill = HILLS[0]

    # Try to detect hill reps
    reps = []
    chart = []
    valley_markers = []
    sawtooth_start_t = 0
    sawtooth_end_t = 0

    try:
        alt_smooth = savgol_filter(alt, window_length=15, polyorder=2)
        saw_start, saw_end = detect_sawtooth_region(alt_smooth)

        # Try to detect which hill
        try:
            detected_hill = detect_hill(df, saw_start, saw_end)
            hill = detected_hill
        except ValueError:
            pass

        df["dist_to_bottom"] = haversine_m(
            df["lat"].values, df["lon"].values, hill["bottom_lat"], hill["bottom_lon"]
        )

        gps_first, gps_last = find_gps_passes(df, saw_start, saw_end)
        valleys = find_elevation_valleys(alt, gps_first, gps_last)

        # Compute stats for all detected rep segments
        all_reps = compute_rep_stats(df, alt, valleys)

        # Filter out reps that don't reach the top
        for rep in all_reps:
            if validate_rep_reaches_top(df, rep["si"], rep["ei"], hill):
                reps.append(rep)

        # Re-number valid reps
        for i, rep in enumerate(reps):
            rep["rep"] = i + 1

        # Build chart data
        step = max(1, len(alt) // 600)
        chart = [
            {"t": float(df["time"].iloc[i]), "a": round(float(alt[i]), 1)}
            for i in range(0, len(alt), step)
        ]
        valley_markers = [
            {"t": float(df["time"].iloc[v]), "a": round(float(alt[v]), 1)}
            for v in valleys
        ]
        sawtooth_start_t = float(df["time"].iloc[saw_start])
        sawtooth_end_t = float(df["time"].iloc[saw_end])

    except (ValueError, IndexError):
        # Hill rep detection failed - build basic chart
        step = max(1, len(alt) // 600)
        chart = [
            {"t": float(df["time"].iloc[i]), "a": round(float(alt[i]), 1)}
            for i in range(0, len(alt), step)
        ]

    # Compute rep summary stats
    rep_total_elevation = sum(r["up_elev"] for r in reps)
    avg_elevation = rep_total_elevation / len(reps) if reps else None
    avg_rep_time = sum(r["full_dur_s"] for r in reps) / len(reps) if reps else None

    return {
        "strava_id": activity.get("id"),
        "strava_name": activity.get("name"),
        "strava_url": f"https://www.strava.com/activities/{activity.get('id')}",
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
            # Rep stats (calculated from stream data)
            "num_reps": len(reps),
            "rep_total_elevation": round(rep_total_elevation, 1),
            "avg_elevation": round(avg_elevation, 1) if avg_elevation is not None else None,
            "avg_rep_time": round(avg_rep_time, 0) if avg_rep_time is not None else None,
            # Strava official stats (what Everesting validates against)
            "strava_elevation_gain": strava_elevation_gain,
            "workout_duration": round(workout_duration, 0),
            "workout_distance": round(workout_distance, 0),
        },
    }


def load_activities():
    """Load activities from the Strava sync file."""
    if not os.path.exists(ACTIVITIES_FILE):
        print(f"Error: {ACTIVITIES_FILE} not found.")
        print("Run 'python strava_sync.py fetch' first.")
        sys.exit(1)

    with open(ACTIVITIES_FILE) as f:
        return json.load(f)


def generate_html(template_path, data, output_path):
    """Inject JSON data into the HTML template."""
    with open(template_path) as f:
        html = f.read()
    html = html.replace("REPLACE_DATA", json.dumps(data))
    html = html.replace("REPLACE_DATE", data["workout_date"])
    with open(output_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Analyze hill reps from Strava activity data.")
    parser.add_argument("--activity", "-a", type=int, help="Analyze specific activity ID")
    parser.add_argument("--output", "-o", default="strava_output", help="Output directory")
    parser.add_argument("--min-elevation", type=float, default=200,
                        help="Minimum elevation gain to analyze (default: 200m)")
    args = parser.parse_args()

    activities = load_activities()
    os.makedirs(args.output, exist_ok=True)

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")

    if args.activity:
        # Analyze specific activity
        activity = next((a for a in activities if a.get("id") == args.activity), None)
        if not activity:
            print(f"Error: Activity {args.activity} not found.")
            sys.exit(1)
        activities_to_analyze = [activity]
    else:
        # Filter activities with sufficient elevation and streams
        activities_to_analyze = [
            a for a in activities
            if a.get("total_elevation_gain", 0) >= args.min_elevation
            and a.get("streams") is not None
        ]

    print(f"Analyzing {len(activities_to_analyze)} activities with ≥{args.min_elevation}m elevation...\n")

    results = []
    everesting_total = 0

    for activity in activities_to_analyze:
        name = activity.get("name", "Unknown")[:30]
        strava_elev = activity.get("total_elevation_gain", 0)
        print(f"  {name}... ", end="")

        result = analyze_strava_activity(activity)
        if result is None:
            print("skipped (no stream data)")
            continue

        num_reps = result["summary"]["num_reps"]
        rep_elev = result["summary"]["rep_total_elevation"]

        if num_reps > 0:
            print(f"{num_reps} reps, {rep_elev:.0f}m (Strava: {strava_elev:.0f}m)")
            everesting_total += strava_elev  # Use Strava's official number

            # Save individual activity JSON
            json_path = os.path.join(args.output, f"activity_{result['strava_id']}.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            # Generate HTML if template exists
            if os.path.isfile(template_path):
                html_path = os.path.join(args.output, f"activity_{result['strava_id']}.html")
                generate_html(template_path, result, html_path)

            results.append(result)
        else:
            print(f"0 reps detected (Strava: {strava_elev:.0f}m)")

    # Summary
    print("\n" + "=" * 60)
    print("EVERESTING PROGRESS (using Strava official elevation)")
    print("=" * 60)

    total_reps = sum(r["summary"]["num_reps"] for r in results)
    total_strava_elev = sum(r["summary"]["strava_elevation_gain"] for r in results)
    everest_height = 8848

    print(f"  Total activities with reps: {len(results)}")
    print(f"  Total reps: {total_reps}")
    print(f"  Total elevation (Strava official): {total_strava_elev:.0f}m")
    print(f"  Progress to Everest: {total_strava_elev / everest_height * 100:.1f}%")
    print(f"  Remaining: {max(0, everest_height - total_strava_elev):.0f}m")

    # Save summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_activities": len(results),
        "total_reps": total_reps,
        "total_strava_elevation": total_strava_elev,
        "everest_progress_percent": round(total_strava_elev / everest_height * 100, 1),
        "remaining_meters": max(0, everest_height - total_strava_elev),
        "activities": [
            {
                "strava_id": r["strava_id"],
                "strava_url": r["strava_url"],
                "name": r["strava_name"],
                "date": r["workout_date"],
                "num_reps": r["summary"]["num_reps"],
                "strava_elevation": r["summary"]["strava_elevation_gain"],
                "rep_elevation": r["summary"]["rep_total_elevation"],
            }
            for r in results
        ]
    }

    summary_path = os.path.join(args.output, "everesting_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved: {summary_path}")
    print(f"  Individual reports: {args.output}/activity_*.json")


if __name__ == "__main__":
    main()
