from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
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
    input_path = None
    output_path = None

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

        # 🎯 פורמט לאורך (9:16)
        output_width = 720
        output_height = 1280

        tracker = None
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except Exception:
            tracker = cv2.legacy.TrackerMOSSE_create()

        tracker.init(frame, (x, y, w, h))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (output_width, output_height)
        )

        # פריים ראשון
        first_frame = frame.copy()
        cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 🔥 חשוב — resize
        first_frame = cv2.resize(first_frame, (output_width, output_height))
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

            # 🔥 זה השינוי הכי חשוב
            output_frame = cv2.resize(output_frame, (output_width, output_height))

            out.write(output_frame)

        cap.release()
        out.release()

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="tracked_clip.mp4"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
