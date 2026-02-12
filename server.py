"""
Flask web server for Hill Rep Analysis.

Usage:
    python server.py

Then open http://localhost:5000 in your browser.
"""

import os
from flask import Flask, jsonify, send_file

from analyze import analyze_fit, get_workout_summary

app = Flask(__name__)

FIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_files")


@app.route("/")
def index():
    """Serve the main HTML page."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")
    return send_file(template_path)


@app.route("/api/workouts")
def list_workouts():
    """List all FIT files with basic metadata, ordered chronologically."""
    workouts = []
    if os.path.isdir(FIT_DIR):
        for filename in os.listdir(FIT_DIR):
            if filename.lower().endswith(".fit"):
                path = os.path.join(FIT_DIR, filename)
                try:
                    summary = get_workout_summary(path)
                    workouts.append(summary)
                except Exception as e:
                    workouts.append({
                        "filename": filename,
                        "date": "Error",
                        "date_display": str(e)[:50],
                    })

    workouts.sort(key=lambda w: w["date"], reverse=True)
    return jsonify(workouts)


@app.route("/api/analyze/<filename>")
def analyze_workout(filename):
    """Run full analysis on a selected workout and return JSON."""
    path = os.path.join(FIT_DIR, filename)

    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        data = analyze_fit(path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"FIT files directory: {FIT_DIR}")
    print("Starting server at http://localhost:5001")
    app.run(debug=True, port=5001)
