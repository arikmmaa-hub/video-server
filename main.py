from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import os
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150MB per chunk

CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"]
}})

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
CHUNKS_FOLDER = "chunks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHUNKS_FOLDER, exist_ok=True)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,X-Requested-With,Accept,Origin")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    return response

# ========== בדיקה שהשרת עובד ==========
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"})

# ========== העלאת חלק מסרטון ==========
@app.route("/upload-chunk", methods=["POST", "OPTIONS"])
def upload_chunk():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    try:
        chunk = request.files.get("chunk")
        upload_id = request.form.get("uploadId")
        chunk_index = request.form.get("chunkIndex")
        if not chunk or not upload_id or chunk_index is None:
            return jsonify({"error": "Missing chunk data"}), 400
        chunk_dir = os.path.join(CHUNKS_FOLDER, upload_id)
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index}")
        chunk.save(chunk_path)
        return jsonify({"status": "ok", "chunkIndex": chunk_index})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== חיבור כל החלקים לסרטון שלם ==========
@app.route("/upload-complete", methods=["POST", "OPTIONS"])
def upload_complete():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    try:
        data = request.get_json()
        upload_id = data.get("uploadId")
        total_chunks = data.get("totalChunks")
        if not upload_id or not total_chunks:
            return jsonify({"error": "Missing data"}), 400
        filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(output_path, "wb") as output_file:
            for i in range(int(total_chunks)):
                chunk_path = os.path.join(CHUNKS_FOLDER, upload_id, f"chunk_{i}")
                with open(chunk_path, "rb") as chunk_file:
                    output_file.write(chunk_file.read())
        chunk_dir = os.path.join(CHUNKS_FOLDER, upload_id)
        for f in os.listdir(chunk_dir):
            os.remove(os.path.join(chunk_dir, f))
        os.rmdir(chunk_dir)
        base_url = request.host_url.rstrip("/")
        video_url = f"{base_url}/uploads/{filename}"
        return jsonify({"url": video_url, "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== הגשת סרטון ==========
@app.route("/uploads/<filename>", methods=["GET"])
def serve_video(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, mimetype="video/mp4")

# ========== עיבוד סרטון עם tracking ==========
@app.route("/process", methods=["POST", "OPTIONS"])
def process_video():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
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
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
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
                target_center_x = last_good_center_x if last_good_center_x else w // 2
            if smooth_center_x is None:
                smooth_center_x = target_center_x
            else:
                alpha = 0.85 if lost_counter < 5 else 0.95
                smooth_center_x = int(alpha * smooth_center_x + (1 - alpha) * target_center_x)
            crop_width = int(h * 9 / 16)
            if crop_width >= w:
                vertical_frame = cv2.resize(frame, (output_width, output_height))
            else:
                x1 = max(0, smooth_center_x - crop_width // 2)
                x2 = min(w, x1 + crop_width)
                cropped = frame[:, x1:x2]
                vertical_frame = cv2.resize(cropped, (output_width, output_height))
            video_writer.write(vertical_frame)
            ret, frame = cap.read()
            if not ret:
                break
        cap.release()
        video_writer.release()
        return send_file(output_path, as_attachment=True, download_name="tracked_clip.mp4", mimetype="video/mp4")
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
