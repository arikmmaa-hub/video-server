from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import subprocess
import tempfile
import os
import cv2

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
cap = cv2.VideoCapture(input_path)

# Tracker
tracker = cv2.TrackerCSRT_create()

ret, frame = cap.read()
if not ret:
    return "Failed to read video", 500

# init tracking
tracker.init(frame, (x, y, w, h))

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, box = tracker.update(frame)
height, width, _ = frames[0].shape

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width, height)
)

for f in frames:
    out.write(f)

out.release()
cap.release()
    if success:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    frames.append(frame)

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
