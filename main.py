from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tempfile
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "server is running"})


def fit_frame_with_padding(frame, target_width=1080, target_height=960):
    h, w = frame.shape[:2]

    scale = min(target_width / w, target_height / h)
    new_width = int(w * scale)
    new_height = int(h * scale)

    resized = cv2.resize(frame, (new_width, new_height))

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    return canvas


@app.route("/process", methods=["POST"])
def process_video():
    input_path = None
    output_path = None
    cap = None
    out = None

    try:
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

        output_path = input_path.replace(".mp4", "_tracked.mp4")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video"}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to read first frame"}), 500

        original_height, original_width = frame.shape[:2]

        x = max(0, min(x, original_width - 1))
        y = max(0, min(y, original_height - 1))
        w = max(1, min(w, original_width - x))
        h = max(1, min(h, original_height - y))

        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except Exception:
            try:
                tracker = cv2.legacy.TrackerMOSSE_create()
            except Exception:
                return jsonify({"error": "No supported OpenCV tracker found"}), 500

        tracker.init(frame, (x, y, w, h))

        output_width = 1080
        output_height = 960

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        if not out.isOpened():
            return jsonify({"error": "Failed to open VideoWriter"}), 500

        first_frame = frame.copy()
        cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        first_frame = fit_frame_with_padding(first_frame, output_width, output_height)
        out.write(first_frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_frame = frame.copy()
            success, box = tracker.update(frame)

            if success:
                bx, by, bw, bh = [int(v) for v in box]
                cv2.rectangle(output_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            output_frame = fit_frame_with_padding(output_frame, output_width, output_height)
            out.write(output_frame)

        cap.release()
        out.release()
        cap = None
        out = None

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="tracked_clip.mp4"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
