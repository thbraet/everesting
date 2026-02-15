"""
Flask web server for Hill Rep Analysis.

Supports both Garmin FIT files and Strava activity data.

Usage:
    python server.py

Then open http://localhost:5001 in your browser.
"""

import json
import os
from datetime import datetime

from flask import Flask, jsonify, send_file

from analyze import analyze_fit, get_workout_summary
from analyze_strava import analyze_strava_activity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIT_DIR = os.path.join(BASE_DIR, "fit_files")
STRAVA_FILE = os.path.join(BASE_DIR, "strava_activities.json")


def load_strava_activities():
    """Load Strava activities from cache file."""
    if not os.path.exists(STRAVA_FILE):
        return []
    with open(STRAVA_FILE) as f:
        return json.load(f)


def get_strava_date(activity):
    """Extract date string from Strava activity."""
    start_date = activity.get("start_date_local", "")
    try:
        date_obj = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        return date_obj.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def build_workout_index():
    """
    Build an index of workouts from both sources, matched by date.

    Returns a list of dicts with:
    - date: YYYY-MM-DD
    - date_display: human readable date
    - garmin_file: filename or None
    - strava_id: activity ID or None
    - strava_name: activity name or None
    """
    workouts = {}

    # Index Garmin FIT files
    if os.path.isdir(FIT_DIR):
        for filename in os.listdir(FIT_DIR):
            if filename.lower().endswith(".fit"):
                path = os.path.join(FIT_DIR, filename)
                try:
                    summary = get_workout_summary(path)
                    date = summary["date"]
                    if date not in workouts:
                        workouts[date] = {
                            "date": date,
                            "date_display": summary["date_display"],
                            "garmin_file": None,
                            "strava_id": None,
                            "strava_name": None,
                        }
                    workouts[date]["garmin_file"] = filename
                except Exception:
                    pass

    # Index Strava activities
    strava_activities = load_strava_activities()
    for activity in strava_activities:
        date = get_strava_date(activity)
        if date:
            if date not in workouts:
                try:
                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                    date_display = date_obj.strftime("%d %b %Y")
                except ValueError:
                    date_display = date
                workouts[date] = {
                    "date": date,
                    "date_display": date_display,
                    "garmin_file": None,
                    "strava_id": None,
                    "strava_name": None,
                }
            workouts[date]["strava_id"] = activity.get("id")
            workouts[date]["strava_name"] = activity.get("name")

    # Sort by date descending
    return sorted(workouts.values(), key=lambda w: w["date"], reverse=True)


@app.route("/")
def index():
    """Serve the main HTML page."""
    template_path = os.path.join(BASE_DIR, "template.html")
    return send_file(template_path)


@app.route("/api/workouts")
def list_workouts():
    """List all workouts from both sources, matched by date."""
    return jsonify(build_workout_index())


@app.route("/api/analyze/garmin/<filename>")
def analyze_garmin(filename):
    """Run full analysis on a Garmin FIT file."""
    path = os.path.join(FIT_DIR, filename)

    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        data = analyze_fit(path)
        data["source"] = "garmin"
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze/strava/<int:activity_id>")
def analyze_strava(activity_id):
    """Run full analysis on a Strava activity."""
    strava_activities = load_strava_activities()
    activity = next((a for a in strava_activities if a.get("id") == activity_id), None)

    if not activity:
        return jsonify({"error": f"Strava activity not found: {activity_id}"}), 404

    # Check if activity has stream data
    if not activity.get("streams"):
        return jsonify({"error": "Activity has no stream data. Re-fetch from Strava."}), 400

    try:
        data = analyze_strava_activity(activity)
        if data is None:
            return jsonify({"error": "Could not analyze activity (insufficient data)"}), 400
        data["source"] = "strava"
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Legacy endpoint for backward compatibility
@app.route("/api/analyze/<filename>")
def analyze_workout_legacy(filename):
    """Legacy endpoint - redirects to Garmin analysis."""
    return analyze_garmin(filename)


if __name__ == "__main__":
    print(f"FIT files directory: {FIT_DIR}")
    print(f"Strava activities file: {STRAVA_FILE}")
    print("Starting server at http://localhost:5001")
    app.run(debug=True, port=5001)
