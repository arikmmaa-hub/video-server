from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import subprocess
import tempfile
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "server is running"})

@app.route("/process", methods=["POST"])
def process_video():
    x = int(request.form.get("x", 0))
y = int(request.form.get("y", 0))
w = int(request.form.get("width", 100))
h = int(request.form.get("height", 100))
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as input_file:
        video_file.save(input_file.name)
        input_path = input_file.name

    output_path = input_path.replace(".mp4", "_vertical.mp4")

    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "crop=ih*9/16:ih,scale=720:1280",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ], check=True)

    os.unlink(input_path)

    return send_file(
        output_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name="vertical_clip.mp4"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
