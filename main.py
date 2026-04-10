from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import os
import uuid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]}})

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"})

@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_video():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400
        
        video_file = request.files["video"]
        filename = f"{uuid.uuid4()}.mp4"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(file_path)
        
        base_url = request.host_url.rstrip("/")
        video_url = f"{base_url}/uploads/{filename}"
        
        response = jsonify({"url": video_url, "filename": filename})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<filename>", methods=["GET"])
def serve_video(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, mimetype="video/mp4")

@app.route("/process", methods=["POST", "OPTIONS"])
def process_video():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response
    
    cap = None
    video_writer = None
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video"]
        x = int(float(request.form["x"]))
        y = int(float(request.form["y"]))
        box_w = int(float(request.form["width"]))
        box_h = int(float(request.form["height"]))

        padding = 0.3
        x = int(x - box_w * padding)
        y = int(y - box_h * padding)
        box_w = int(box_w * (1 + padding * 2))
        box_h = int(box_h * (1 + padding * 2))

        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
        output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.mp4")
        video_file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        output_width = 1080
        output_height = 1920
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (output_width, output_height)
        )

        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to read video"}), 400

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x, y, box_w, box_h))

        smooth_center_x = None
        last_good_center_x = None
        lost_counter = 0

        while True:
            success, box = tracker.update(frame)
            h, w = frame.shape[:2]

            if success:
                tx, ty, tw, th = [int(v) for v in box]
                target_center_x = tx + tw // 2
                last_good_center_x = target_center_x
                lost_counter = 0
            else:
                lost_counter += 1
                if last_good_center_x is not None:
                    target_center_x = last_good_center_x
                else:
                    target_center_x = w // 2

            if smooth_center_x is None:
                smooth_center_x = target_center_x
            else:
                alpha = 0.85 if lost_counter < 5 else 0.95
                smooth_center_x = int(alpha * smooth_center_x + (1 - alpha) * target_center_x)

            crop_width = int(h * 9 / 16)
            if crop_width >= w:
                vertical_frame = cv2.resize(frame, (output_width, output_height))
            else:
                x1 = smooth_center_x - crop_width // 2
                x2 = x1 + crop_width
                x1 = max(0, x1)
                x2 = min(w, x2)
                cropped = frame[:, x1:x2]
                vertical_frame = cv2.resize(cropped, (output_width, output_height))

            video_writer.write(vertical_frame)
            ret, frame = cap.read()
            if not ret:
                break

        cap.release()
        video_writer.release()

        response = send_file(
            output_path,
            as_attachment=True,
            download_name="tracked_clip.mp4",
            mimetype="video/mp4"
        )
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cap:
            cap.release()
        if video_writer:
            video_writer.release()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
