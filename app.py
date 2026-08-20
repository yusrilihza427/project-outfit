from flask import Flask, request, jsonify, render_template, send_from_directory
from utils.recommendation import RecommendationEngine
import os

app = Flask(__name__)

# Load model sekali saat aplikasi dijalankan
engine = RecommendationEngine()


# ==========================
# HOME
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# MENAMPILKAN GAMBAR DATASET
# ==========================
@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory("dataset", filename)


# ==========================
# RECOMMENDATION API
# ==========================
@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.get_json()

    user_input = {
        "aktivitas": data["aktivitas"],
        "undertone": data["undertone"],
        "hijab": data["hijab"]
    }

    results = engine.recommend(
        user_input,
        top_k=5
    )

    print(results)

    return jsonify(results)


# ==========================
# RUN FLASK
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )