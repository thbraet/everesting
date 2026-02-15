"""
Static site builder for Hill Rep Analysis.

Generates a dist/ folder with all pre-computed workout data that can be
deployed to GitHub Pages, Netlify, or any static hosting.

Usage:
    python build.py

Output:
    dist/
    ├── index.html          # Main page
    ├── workouts.json       # List of all workouts
    └── data/
        └── <filename>.json # Analysis data for each workout
"""

import json
import os
import shutil

from analyze import analyze_fit, get_workout_summary

FIT_DIR = "fit_files"
DIST_DIR = "dist"
DATA_DIR = os.path.join(DIST_DIR, "data")


def build():
    # Clean and create dist directory
    if os.path.exists(DIST_DIR):
        shutil.rmtree(DIST_DIR)
    os.makedirs(DATA_DIR)

    # Find all FIT files
    fit_files = []
    if os.path.isdir(FIT_DIR):
        fit_files = [f for f in os.listdir(FIT_DIR) if f.lower().endswith(".fit")]

    print(f"Found {len(fit_files)} FIT files")

    # Process each workout
    workouts = []
    errors = 0
    for filename in fit_files:
        path = os.path.join(FIT_DIR, filename)
        print(f"  Processing {filename}...")

        try:
            # Run full analysis
            data = analyze_fit(path)

            # Save analysis as JSON (even if 0 reps, for workout stats)
            json_filename = filename.rsplit(".", 1)[0] + ".json"
            json_path = os.path.join(DATA_DIR, json_filename)
            with open(json_path, "w") as f:
                json.dump(data, f)

            summary = data["summary"]

            # Extract summary for workout list
            workouts.append({
                "filename": filename,
                "date": data["workout_date"],
                "date_sort": get_workout_summary(path)["date"],  # ISO format for sorting
                "hill_id": data["hill"]["id"],
                "hill_name": data["hill"]["name"],
                # Rep stats (may be None/0 if no reps)
                "num_reps": summary["num_reps"],
                "rep_total_elevation": summary["rep_total_elevation"],
                "avg_elevation": summary["avg_elevation"],
                "avg_rep_time": summary["avg_rep_time"],
                # Workout stats
                "workout_duration": summary["workout_duration"],
                "workout_distance": summary["workout_distance"],
                "workout_total_ascent": summary["workout_total_ascent"],
            })

            if summary["num_reps"] > 0:
                print(f"    → {summary['num_reps']} reps @ {data['hill']['name']}")
            else:
                print(f"    → No hill reps (workout only)")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            errors += 1

    # Sort workouts by date (newest first)
    workouts.sort(key=lambda w: w["date_sort"], reverse=True)

    # Write workouts list
    workouts_path = os.path.join(DIST_DIR, "workouts.json")
    with open(workouts_path, "w") as f:
        json.dump(workouts, f, indent=2)
    print(f"\nWorkouts index: {workouts_path}")

    # Copy and modify HTML template
    with open("template_static.html") as f:
        html = f.read()

    html_path = os.path.join(DIST_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML page: {html_path}")

    reps_workouts = sum(1 for w in workouts if w["num_reps"] > 0)
    print(f"\n✓ Build complete! Output in {DIST_DIR}/")
    print(f"  {len(workouts)} workouts total ({reps_workouts} with hill reps)")
    if errors > 0:
        print(f"  {errors} files failed to process")
    print(f"  To preview locally: cd {DIST_DIR} && python3 -m http.server 8000")
    print(f"  Then open: http://localhost:8000")


if __name__ == "__main__":
    build()
