from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import os
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"})


@app.route("/process", methods=["POST"])
def process_video():
    cap = None
    video_writer = None

    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        if "x" not in request.form or "y" not in request.form or "width" not in request.form or "height" not in request.form:
            return jsonify({"error": "Missing tracking box data"}), 400

        video_file = request.files["video"]

        x = int(float(request.form["x"]))
        y = int(float(request.form["y"]))
        box_w = int(float(request.form["width"]))
        box_h = int(float(request.form["height"]))

        input_filename = f"{uuid.uuid4()}.mp4"
        output_filename = f"{uuid.uuid4()}.mp4"

        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        video_file.save(input_path)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return jsonify({"error": "Could not open input video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        output_width = 1080
        output_height = 1920

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (output_width, output_height)
        )

        if not video_writer.isOpened():
            cap.release()
            return jsonify({"error": "VideoWriter failed to open"}), 500

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            video_writer.release()
            return jsonify({"error": "Could not read first frame"}), 400

        tracker = cv2.TrackerCSRT_create()
        tracker.init(first_frame, (x, y, box_w, box_h))

        current_frame = first_frame

        while True:
            frame = current_frame
            if frame is None:
                break

            success, box = tracker.update(frame)

            h, w = frame.shape[:2]

            if success:
                tx, ty, tw, th = [int(v) for v in box]
                ball_center_x = tx + tw // 2
            else:
                ball_center_x = w // 2

            crop_width = int(h * 9 / 16)

            if crop_width > w:
                vertical_frame = cv2.resize(frame, (output_width, output_height))
            else:
                x1 = ball_center_x - crop_width // 2
                x2 = x1 + crop_width

                if x1 < 0:
                    x1 = 0
                    x2 = crop_width

                if x2 > w:
                    x2 = w
                    x1 = w - crop_width

                cropped = frame[:, x1:x2]
                vertical_frame = cv2.resize(cropped, (output_width, output_height))

            video_writer.write(vertical_frame)

            ret, current_frame = cap.read()
            if not ret:
                break

        cap.release()
        video_writer.release()

        if not os.path.exists(output_path):
            return jsonify({"error": "Output file was not created"}), 500

        if os.path.getsize(output_path) == 0:
            return jsonify({"error": "Output file is empty"}), 500

        return send_file(
            output_path,
            as_attachment=True,
            download_name="tracked_clip.mp4",
            mimetype="video/mp4"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
