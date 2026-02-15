#!/usr/bin/env python3
"""
Strava API integration for fetching activities with detailed streams.

Usage:
    python strava_sync.py auth      # First-time authentication
    python strava_sync.py fetch     # Fetch activities since Jan 19, 2026
    python strava_sync.py fetch --since 2026-01-19  # Custom date
"""

import os
import sys
import json
import time
import argparse
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import requests

# Load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

CLIENT_ID = os.environ.get('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.environ.get('STRAVA_CLIENT_SECRET')
TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'strava_tokens.json')
ACTIVITIES_FILE = os.path.join(os.path.dirname(__file__), 'strava_activities.json')

AUTHORIZE_URL = 'https://www.strava.com/oauth/authorize'
TOKEN_URL = 'https://www.strava.com/oauth/token'
API_BASE = 'https://www.strava.com/api/v3'

# OAuth callback handler
auth_code = None

class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        query = parse_qs(urlparse(self.path).query)

        if 'code' in query:
            auth_code = query['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''
                <html><body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                <h1>Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                </body></html>
            ''')
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            error = query.get('error', ['Unknown error'])[0]
            self.wfile.write(f'<html><body><h1>Error: {error}</h1></body></html>'.encode())

    def log_message(self, format, *args):
        pass  # Suppress server logs


def check_credentials():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: Missing Strava credentials!")
        print()
        print("Please create a .env file with:")
        print("  STRAVA_CLIENT_ID=your_client_id")
        print("  STRAVA_CLIENT_SECRET=your_client_secret")
        print()
        print("Get these from: https://www.strava.com/settings/api")
        sys.exit(1)


def authenticate():
    """Run OAuth flow to get access token."""
    check_credentials()

    # Start local server for OAuth callback
    port = 8000
    server = HTTPServer(('localhost', port), OAuthHandler)

    # Build authorization URL
    redirect_uri = f'http://localhost:{port}/callback'
    auth_url = (
        f"{AUTHORIZE_URL}?"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope=activity:read_all"
    )

    print("Opening browser for Strava authorization...")
    print(f"If browser doesn't open, visit: {auth_url}")
    webbrowser.open(auth_url)

    print("Waiting for authorization...")
    server.handle_request()

    if not auth_code:
        print("Error: No authorization code received")
        sys.exit(1)

    # Exchange code for tokens
    print("Exchanging code for access token...")
    response = requests.post(TOKEN_URL, data={
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': auth_code,
        'grant_type': 'authorization_code'
    })

    if response.status_code != 200:
        print(f"Error getting token: {response.text}")
        sys.exit(1)

    tokens = response.json()
    save_tokens(tokens)
    print(f"Authentication successful! Logged in as: {tokens['athlete']['firstname']} {tokens['athlete']['lastname']}")


def save_tokens(tokens):
    """Save tokens to file."""
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f, indent=2)


def load_tokens():
    """Load tokens from file."""
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE) as f:
        return json.load(f)


def refresh_token_if_needed(tokens):
    """Refresh access token if expired."""
    if tokens['expires_at'] < time.time():
        print("Access token expired, refreshing...")
        response = requests.post(TOKEN_URL, data={
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'grant_type': 'refresh_token',
            'refresh_token': tokens['refresh_token']
        })

        if response.status_code != 200:
            print(f"Error refreshing token: {response.text}")
            print("Please re-authenticate with: python strava_sync.py auth")
            sys.exit(1)

        tokens = response.json()
        save_tokens(tokens)
        print("Token refreshed successfully")

    return tokens


def fetch_activity_streams(activity_id, access_token):
    """Fetch detailed streams (elevation profile, etc.) for an activity."""
    stream_types = ['time', 'distance', 'altitude', 'heartrate', 'latlng']

    response = requests.get(
        f"{API_BASE}/activities/{activity_id}/streams",
        headers={'Authorization': f'Bearer {access_token}'},
        params={
            'keys': ','.join(stream_types),
            'key_type': 'time'
        }
    )

    if response.status_code == 200:
        streams = response.json()
        # Convert list of stream objects to dict keyed by type
        return {s['type']: s['data'] for s in streams}
    elif response.status_code == 404:
        # Some activities (manual entries) don't have streams
        return None
    else:
        print(f"  Warning: Could not fetch streams for activity {activity_id}: {response.status_code}")
        return None


def fetch_activities(since_date):
    """Fetch all activities since the given date with detailed streams."""
    check_credentials()

    tokens = load_tokens()
    if not tokens:
        print("Not authenticated. Run: python strava_sync.py auth")
        sys.exit(1)

    tokens = refresh_token_if_needed(tokens)
    access_token = tokens['access_token']

    # Convert date to epoch timestamp
    since_ts = int(datetime.strptime(since_date, '%Y-%m-%d').timestamp())

    print(f"Fetching activity list since {since_date}...")

    activities = []
    page = 1
    per_page = 100

    while True:
        response = requests.get(
            f"{API_BASE}/athlete/activities",
            headers={'Authorization': f'Bearer {access_token}'},
            params={
                'after': since_ts,
                'page': page,
                'per_page': per_page
            }
        )

        if response.status_code != 200:
            print(f"Error fetching activities: {response.text}")
            sys.exit(1)

        batch = response.json()
        if not batch:
            break

        activities.extend(batch)
        print(f"  Fetched page {page}: {len(batch)} activities")
        page += 1

    print(f"Total activities found: {len(activities)}")

    # Fetch detailed streams for each activity
    print(f"\nFetching detailed streams for each activity...")
    for i, act in enumerate(activities):
        activity_id = act['id']
        name = act.get('name', 'Unknown')[:30]
        print(f"  [{i+1}/{len(activities)}] {name}...", end=' ')

        streams = fetch_activity_streams(activity_id, access_token)
        if streams:
            act['streams'] = streams
            print(f"OK ({len(streams.get('altitude', []))} points)")
        else:
            act['streams'] = None
            print("No streams")

        # Rate limiting: Strava API allows 100 requests per 15 minutes
        # Be conservative with a small delay
        time.sleep(0.2)

    # Save to file
    with open(ACTIVITIES_FILE, 'w') as f:
        json.dump(activities, f, indent=2)

    print(f"\nActivities with streams saved to: {ACTIVITIES_FILE}")

    # Print summary
    print("\nActivity Summary:")
    print("-" * 70)
    for act in activities:
        date = act['start_date_local'][:10]
        name = act['name'][:25]
        elev = act.get('total_elevation_gain', 0)
        dist = act.get('distance', 0) / 1000
        act_type = act.get('type', 'Unknown')
        has_streams = "Yes" if act.get('streams') else "No"
        print(f"{date} | {act_type:10} | {elev:6.0f}m | {dist:5.1f}km | {has_streams:3} | {name}")

    return activities


def main():
    parser = argparse.ArgumentParser(description='Strava activity sync')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Auth command
    subparsers.add_parser('auth', help='Authenticate with Strava')

    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch activities')
    fetch_parser.add_argument(
        '--since',
        default='2026-01-19',
        help='Fetch activities since this date (YYYY-MM-DD). Default: 2026-01-19'
    )

    args = parser.parse_args()

    if args.command == 'auth':
        authenticate()
    elif args.command == 'fetch':
        fetch_activities(args.since)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
