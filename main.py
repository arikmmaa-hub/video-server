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
    cap = None
    out = None

    try:
        x = int(request.form.get("x", 0))
        y = int(request.form.get("y", 0))
        w = int(request.form.get("width", 100))
        h = int(request.form.get("height", 100))

        print("=== PROCESS STARTED ===")
        print(f"Tracking box: x={x}, y={y}, w={w}, h={h}")

        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video"]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as input_file:
            video_file.save(input_file.name)
            input_path = input_file.name

        output_path = input_path.replace(".mp4", "_tracked.mp4")

        cap = cv2.VideoCapture(input_path)
        print("cap opened:", cap.isOpened())

        if not cap.isOpened():
            return jsonify({"error": "Failed to open video"}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        ret, frame = cap.read()
        print("first frame read:", ret)

        if not ret:
            return jsonify({"error": "Failed to read first frame"}), 500

        print("first frame shape:", frame.shape)
        print("first frame mean:", float(frame.mean()))

        height, width, _ = frame.shape

        # מוודא שהריבוע לא יוצא מגבולות הפריים
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        print(f"Adjusted box: x={x}, y={y}, w={w}, h={h}")
        print(f"Video size: width={width}, height={height}, fps={fps}")

        tracker = None
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
            print("Using CSRT tracker")
        except Exception:
            try:
                tracker = cv2.legacy.TrackerMOSSE_create()
                print("Using MOSSE tracker")
            except Exception:
                return jsonify({"error": "No supported OpenCV tracker found"}), 500

        tracker.init(frame, (x, y, w, h))

        # כותבים ישר ל-MP4 בלי AVI ובלי ffmpeg
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print("writer opened:", out.isOpened())

        if not out.isOpened():
            return jsonify({"error": "Failed to open VideoWriter"}), 500

        # פריים ראשון
        first_frame = frame.copy()
        cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(first_frame)

        frame_count = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                print("Skipped None frame")
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            output_frame = frame.copy()
            success, box = tracker.update(frame)

            if success:
                bx, by, bw, bh = [int(v) for v in box]
                cv2.rectangle(output_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            if frame_count % 30 == 0:
                print(f"frame {frame_count}, mean={float(output_frame.mean())}")

            out.write(output_frame)
            frame_count += 1

        print("total frames written:", frame_count)

        cap.release()
        out.release()
        cap = None
        out = None

        if not os.path.exists(output_path):
            return jsonify({"error": "Output file was not created"}), 500

        file_size = os.path.getsize(output_path)
        print("output file size:", file_size)

        if file_size == 0:
            return jsonify({"error": "Output file is empty"}), 500

        print("=== PROCESS FINISHED SUCCESSFULLY ===")

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="tracked_clip.mp4"
        )

    except Exception as e:
        print("SERVER ERROR:", str(e))
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
