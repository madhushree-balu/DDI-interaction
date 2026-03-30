"""
app.py  (project root)
======================
Flask application factory. Run with:

    flask --app app run --debug --port 5000

or:

    python app.py
"""

import os
import tempfile
from flask import Flask, request, jsonify, render_template

from blueprint import api_bp, init_analyser
import ocr

def create_app() -> Flask:
    app = Flask(__name__)

    # Load datasets and initialise the analyser once at startup.
    # Override the data directory via the POLYGUARD_DATA_DIR env var.
    init_analyser(data_dir="./datasets")

    # All API endpoints live under /api
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/ocr", methods=["POST"])
    def api_ocr():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        try:
            import numpy as np
            import cv2
            
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({"error": "Invalid or unreadable image format."}), 400

            processed_img = ocr.preprocess_image(img)
            raw_text = ocr.extract_text(processed_img)
            result = ocr.analyze_with_gemini(raw_text)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=80)