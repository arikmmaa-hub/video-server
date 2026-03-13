from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import subprocess
import tempfile
import os

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video_file = request.files['video']
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as input_file:
        video_file.save(input_file.name)
        input_path = input_file.name
    
    output_path = input_path.replace('.mp4', '_vertical.mp4')
    
    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-vf', 'crop=ih*(9/16):ih,scale=1080:1920',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',
        output_path
    ], check=True)
    
    os.unlink(input_path)
    
    return send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name='vertical_clip.mp4')

if __name__ == '__main__':
port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
