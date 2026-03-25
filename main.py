from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import uuid
import math

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# YOLO11 נתמך ב-Ultralytics, וקובצי משקל כמו yolo11n.pt הם פורמט תקין. 
# אם תרצה בהמשך, אפשר להחליף ל-yolo11s.pt לדיוק טוב יותר.
model = YOLO("yolo11n.pt")


def get_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def get_center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def center_distance(box1, box2):
    c1x, c1y = get_center(box1)
    c2x, c2y = get_center(box2)
    return math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)


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

        selected_box = (x, y, box_w, box_h)

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

        tracked_box = selected_box
        smooth_center_x = None
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                continue

            h, w = frame.shape[:2]

            # YOLO כל 3 פריימים כדי לא להעמיס מדי
            if frame_index % 3 == 0:
                results = model(frame, verbose=False)

                detections = []
                if len(results) > 0:
                    result = results[0]

                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())

                            # class 0 = person ב-COCO
                            if cls_id != 0:
                                continue

                            if conf < 0.35:
                                continue

                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            px = int(x1)
                            py = int(y1)
                            pw = int(x2 - x1)
                            ph = int(y2 - y1)

                            detections.append((px, py, pw, ph))

                if detections:
                    best_box = None
                    best_score = -1

                    for det in detections:
                        iou = get_iou(tracked_box, det)
                        dist = center_distance(tracked_box, det)

                        # ציון משולב: חפיפה + קרבה
                        score = (iou * 3.0) - (dist / max(w, 1))

                        if score > best_score:
                            best_score = score
                            best_box = det

                    if best_box is not None:
                        tracked_box = best_box

            tx, ty, tw, th = tracked_box
            player_center_x = tx + tw // 2

            if smooth_center_x is None:
                smooth_center_x = player_center_x
            else:
                smooth_center_x = int(0.85 * smooth_center_x + 0.15 * player_center_x)

            crop_width = int(h * 9 / 16)

            if crop_width >= w:
                vertical_frame = cv2.resize(frame, (output_width, output_height))
            else:
                x1 = smooth_center_x - crop_width // 2
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
            frame_index += 1

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
