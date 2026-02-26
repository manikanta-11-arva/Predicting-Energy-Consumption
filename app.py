"""
app.py — Flask backend for AI Energy Consumption Predictor
────────────────────────────────────────────────────────────
Routes:
  GET  /           → Serve index.html
  POST /upload     → Accept CSV, run ML pipeline, return JSON
  GET  /download   → Download last prediction results as CSV
"""

import os, io, csv, json
from flask      import Flask, request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename
from ml_model   import run_pipeline

# ─── FLASK CONFIG ────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "energy-ai-secret-2024"

UPLOAD_FOLDER   = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTS    = {"csv"}
MAX_CONTENT_MB  = 16

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store for last result (used by /download)
_last_result: dict = {}


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main single-page application."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a CSV file upload, run the full ML pipeline, and return results as JSON.
    Handles: missing file, wrong extension, validation errors, ML errors.
    """
    global _last_result

    # 1. Check file is present in the request
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file supplied. Please select a CSV file."}), 400

    file = request.files["file"]

    # 2. Check a filename was actually provided
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected."}), 400

    # 3. Validate extension
    if not allowed_file(file.filename):
        return jsonify({
            "status" : "error",
            "message": "Invalid file type. Only .csv files are accepted."
        }), 400

    # 4. Save uploaded file safely
    filename    = secure_filename(file.filename)
    save_path   = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # 5. Run ML pipeline — catch user-friendly errors vs unexpected crashes
    try:
        result = run_pipeline(save_path)
    except ValueError as ve:
        # User-facing validation errors (missing columns, too small, etc.)
        return jsonify({"status": "error", "message": str(ve)}), 422
    except Exception as ex:
        # Unexpected server errors
        return jsonify({
            "status" : "error",
            "message": f"Processing failed: {str(ex)}"
        }), 500
    finally:
        # Always clean up the uploaded file
        if os.path.exists(save_path):
            os.remove(save_path)

    # 6. Cache result for CSV download
    _last_result = result

    return jsonify(result), 200


@app.route("/download")
def download():
    """
    Stream the last prediction result (7-day forecast + anomalies) as a CSV download.
    """
    if not _last_result:
        return jsonify({"status": "error", "message": "No prediction results available yet."}), 404

    output = io.StringIO()
    writer = csv.writer(output)

    # ── 7-Day Forecast sheet ──────────────────────────────────────────────────
    writer.writerow(["=== 7-Day Energy Forecast ==="])
    writer.writerow(["Date", "Predicted KWH", "Model"])
    best = _last_result.get("best_model", "N/A")
    for row in _last_result.get("forecast", []):
        writer.writerow([row["date"], row["kwh"], best])

    writer.writerow([])  # blank separator

    # ── Model Metrics ─────────────────────────────────────────────────────────
    writer.writerow(["=== Model Performance ==="])
    writer.writerow(["Model", "MAE", "RMSE", "R2 Score"])
    for m in _last_result.get("models", []):
        writer.writerow([m["model"], round(m["mae"],2), round(m["rmse"],2), round(m["r2"],4)])

    writer.writerow([])

    # ── Anomalies ─────────────────────────────────────────────────────────────
    anom = _last_result.get("anomalies", {})
    writer.writerow(["=== Detected Anomalies ==="])
    writer.writerow(["Date", "KWH", "Type"])
    for a in anom.get("anomalies", []):
        writer.writerow([a["date"], a["kwh"], a["type"]])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="energy_prediction_results.csv",
    )


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Energy Consumption Predictor — Flask Server")
    print("  → http://127.0.0.1:8080")
    print("=" * 60)
    app.run(debug=True, port=8080)
