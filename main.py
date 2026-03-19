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
            if os.path.exists(input_path):
                os.unlink(input_path)
            return jsonify({"error": "Failed to open video"}), 500

        ret, frame = cap.read()
        if not ret:
            cap.release()
            if os.path.exists(input_path):
                os.unlink(input_path)
            return jsonify({"error": "Failed to read first frame"}), 500

        # נסה CSRT דרך legacy, ואם אין - עבור ל-MOSSE
        tracker = None
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except Exception:
            try:
                tracker = cv2.legacy.TrackerMOSSE_create()
            except Exception:
                cap.release()
                if os.path.exists(input_path):
                    os.unlink(input_path)
                return jsonify({"error": "No supported OpenCV tracker found"}), 500

        tracker.init(frame, (x, y, w, h))

        frames = [frame]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            success, box = tracker.update(frame)

            if success:
                bx, by, bw, bh = [int(v) for v in box]
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            frames.append(frame)

        cap.release()

        if not frames:
            if os.path.exists(input_path):
                os.unlink(input_path)
            return jsonify({"error": "No frames processed"}), 500

        height, width, _ = frames[0].shape

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (width, height)
        )

        for f in frames:
            out.write(f)

        out.release()

        if os.path.exists(input_path):
            os.unlink(input_path)

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="tracked_clip.mp4"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
