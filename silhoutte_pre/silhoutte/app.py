from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import os
import threading
from datetime import datetime
from werkzeug.utils import secure_filename
import shutil
from silhoutte_pre.silhoutte.scripts.sil_vid import process_video
from opengait.register_owner import register_owner_from_frames


app = Flask(__name__)
app.secret_key = os.urandom(24)  
camera = None
is_recording = False
video_writer = None
current_video_path = None
record_lock = threading.Lock()

def check_auth():
    return 'username' in session

def get_camera():
    global camera
    if camera is None:
        print(" Initializing camera...")
        camera = cv2.VideoCapture(0)  
        if not camera.isOpened():
            print("Camera at index 0 not found. Trying index 1...")
            camera = cv2.VideoCapture(1)
            if not camera.isOpened():
                print("No available camera found.")
                camera = None
    return camera


def generate_frames():
    """Stream frames from webcam and save if recording"""
    global is_recording, video_writer, current_video_path
    camera = get_camera()

    if camera is None:
        print("generate_frames called but camera is not available.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        if is_recording:
            with record_lock:
                if video_writer is None and current_video_path:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(current_video_path, fourcc, 20.0, (width, height))
                    print(f"🎥 Started saving video to: {current_video_path}")
                if video_writer is not None:
                    video_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    if not check_auth():
        return redirect(url_for('register'))
    return render_template('index.html', username=session.get('username'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if not check_auth():
        return redirect(url_for('register'))

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    username = session.get('username', 'unknown')
    view_angle = request.form.get('view', '0')
    if view_angle not in ['0', '45', '90', '135', '180']:
        return jsonify({"status": "error", "message": "Invalid angle"}), 400

    base_dir = os.path.join('recordings', username, view_angle)
    os.makedirs(base_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(base_dir) if f.endswith(".avi")]
    indices = []
    for f in existing_files:
        parts = f.split("-")
        if len(parts) >= 3 and parts[1] == "nm":
            try:
                indices.append(int(parts[2]))
            except:
                pass
    next_index = max(indices, default=0) + 1
    index_str = str(next_index).zfill(2)

    filename = f"{username}-nm-{index_str}-{view_angle}.avi"
    filepath = os.path.join(base_dir, filename)
    file.save(filepath)

    output_base = os.path.join("processed", username, view_angle)
    frames_folder = os.path.join(output_base, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    processed_video = os.path.join(output_base, filename.replace(".avi", "_silhouette.mp4"))
    process_video(filepath,
                output_video=processed_video,
                output_folder=frames_folder,
                box_size=(128, 128))
    print(f"Preprocessing done. Silhouette video at {processed_video}")
    
    cfg_path = "opengait/configs/gaitpart/gaitpart.yaml"
    model_ckpt = "opengait/output/CASIA-B/GaitPart/GaitPart/checkpoints/GaitPart-120000.pt"
        
    register_owner_from_frames (
        username=username,
        view=view_angle,
        frames_folder=frames_folder,
        cfg_path=cfg_path,
        model_ckpt=model_ckpt
    )
        
    return jsonify({"status": "success", "message": f"Uploaded video saved at {filepath}"})


@app.route('/video_feed')
def video_feed():
    if not check_auth():
        return redirect(url_for('register'))

    if get_camera() is None:
        return "Camera not available", 500

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Simple registration (no DB, just session)"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match!")

        session['username'] = username
        session['email'] = email
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple login (just matches session values)"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')

        if session.get('username') == username and session.get('email') == email:
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid login (session-based only)")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('register'))


@app.route('/start_recording')
def start_recording():
    """Start saving a video into user/angle subfolders with incrementing filename"""
    if not check_auth():
        return redirect(url_for('register'))

    global is_recording, current_video_path, video_writer

    username = session.get('username', 'unknown')
    view_angle = request.args.get('view', '0')
    if view_angle not in ['0', '45', '90', '135', '180']:
        return jsonify({"status": "error", "message": "Invalid angle selected"}), 400

    base_dir = os.path.join('recordings', username, view_angle)
    os.makedirs(base_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(base_dir) if f.endswith(".avi")]
    indices = []
    for f in existing_files:
        parts = f.split("-")
        if len(parts) >= 3 and parts[1] == "nm":
            try:
                indices.append(int(parts[2]))
            except:
                pass
    next_index = max(indices, default=0) + 1
    index_str = str(next_index).zfill(2)  


    filename = f"{username}-nm-{index_str}-{view_angle}.avi"
    current_video_path = os.path.join(base_dir, filename)

    is_recording = True
    video_writer = None  

    print(f"Recording started for {username}, angle={view_angle}°, saved at {current_video_path}")
    return jsonify({"status": "success", "message": f"Recording started for {view_angle}° view"})


@app.route('/stop_recording')
def stop_recording():
    if not check_auth():
        return redirect(url_for('register'))

    global is_recording, video_writer, current_video_path
    is_recording = False

    with record_lock:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved at {current_video_path}")
            video_writer = None
        saved_path = current_video_path
        current_video_path = None

    if saved_path:
        username = session.get('username', 'unknown')
        filename = os.path.basename(saved_path)  
        parts = filename.split("-")
        angle = parts[-1].replace(".avi", "") if len(parts) >= 4 else "0"
        output_base = os.path.join("processed", username, angle)
        frames_folder = os.path.join(output_base, "frames")
        os.makedirs(frames_folder, exist_ok=True)

        processed_video = os.path.join(output_base, filename.replace(".avi", "_silhouette.mp4"))

        process_video(saved_path,
                      output_video=processed_video,
                      output_folder=frames_folder,
                      box_size=(128, 128))
        
        cfg_path = "opengait/configs/gaitpart/gaitpart.yaml"
        model_ckpt = "opengait/output/CASIA-B/GaitPart/GaitPart/checkpoints/GaitPart-120000.pt"
        
        register_owner_from_frames (
            username=username,
            view=angle,
            frames_folder=frames_folder,
            cfg_path=cfg_path,
            model_ckpt=model_ckpt
        )
        
    return jsonify({
        "status": "success",
        "message": f"Recording stopped. Saved at {saved_path}, silhouettes extracted!"
    })


if __name__ == '__main__':
    app.run(debug=True)

