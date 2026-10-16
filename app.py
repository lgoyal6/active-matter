"""Serve the public experiment and the unchanged recorded-study explorer.
Only whitelisted demo assets are served; the API accepts bounded JSON inputs.
No accounts, external services, uploads or persistent writes are used.
"""
from pathlib import Path
import json

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from live_demo import run_experiment

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096
DEMO = Path(__file__).with_name("demo")
ASSETS = {"app.js", "style.css", "data.json", "live.js", "live.css"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field.")
        result[key] = value
    return result


@app.after_request
def headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return response


@app.get("/")
def home():
    return send_from_directory(DEMO, "live.html")


@app.get("/study")
def study():
    return send_from_directory(DEMO, "index.html")


@app.get("/<name>")
def asset(name):
    if name not in ASSETS:
        return jsonify(error="Page not found."), 404
    return send_from_directory(DEMO, name)


@app.post("/api/simulate")
def simulate_request():
    if not request.is_json:
        return jsonify(error="Send application/json."), 415
    try:
        payload = json.loads(request.get_data(), object_pairs_hook=unique_object)
        return jsonify(run_experiment(payload))
    except (ValueError, BadRequest, RecursionError) as error:
        message = str(error) if isinstance(error, ValueError) else "Malformed JSON."
        return jsonify(error=message), 400
    except RuntimeError:
        return jsonify(error="The model could not produce a valid trajectory."), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="Request exceeds 4 KiB."), 413
