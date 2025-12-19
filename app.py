import os
import io
import threading
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file, abort
from pymongo import MongoClient
from bson.objectid import ObjectId
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

MONGODB_DB = os.environ.get("MONGODB_DB", "attendance_system")

app = Flask(__name__, static_folder="static", template_folder="templates")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

def init_db():
    students_col = db["students"]
    attendance_col = db["attendance"]
    students_col.create_index("created_at")
    attendance_col.create_index("timestamp")
    attendance_col.create_index("student_id")

init_db()

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

# ensure initial train status file exists
write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    print("TEST ROUTE HIT", flush=True)
    return jsonify({"status": "test working"})

# Dashboard simple API for attendance stats (last 30 days)
@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    records = list(db["attendance"].find({}, {"timestamp": 1}))
    if not records:
        from datetime import date, timedelta
        days = [(date.today() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0]*30})
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [ (datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1) ]
    counts = [ int(df[df['date'] == d].shape[0]) for d in last_30 ]
    dates = [ d.strftime("%d-%b") for d in last_30 ]
    return jsonify({"dates": dates, "counts": counts})

# -------- Add student (form) --------
@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
    # POST: save student metadata and return student_id
    data = request.form
    name = data.get("name","").strip()
    roll = data.get("roll","").strip()
    cls = data.get("class","").strip()
    sec = data.get("sec","").strip()
    reg_no = data.get("reg_no","").strip()
    if not name:
        return jsonify({"error":"name required"}), 400
    now = datetime.datetime.utcnow().isoformat()
    student_doc = {
        "name": name,
        "roll": roll,
        "class": cls,
        "section": sec,
        "reg_no": reg_no,
        "created_at": now
    }
    result = db["students"].insert_one(student_doc)
    sid = str(result.inserted_id)
    # create dataset folder for this student
    os.makedirs(os.path.join(DATASET_DIR, sid), exist_ok=True)
    return jsonify({"student_id": sid})

# -------- Upload face images (after capture) --------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.utcnow().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})

# -------- Train model (start background thread) --------
@app.route("/train_model", methods=["GET"])
def train_model_route():
    import sys
    sys.stdout.write("=== TRAIN ROUTE CALLED ===\n")
    sys.stdout.flush()
    
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status":"already_running"}), 202
    
    sys.stdout.write(f"DATASET_DIR: {DATASET_DIR}\n")
    sys.stdout.flush()
    
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    
    def callback(p, m):
        sys.stdout.write(f"[CALLBACK] {p}% - {m}\n")
        sys.stdout.flush()
        write_train_status({"running": True, "progress": p, "message": m})
    
    def train_wrapper():
        import sys
        log_path = os.path.join(APP_DIR, "thread_debug.log")
        try:
            sys.stdout.write("[THREAD] Training thread started\n")
            sys.stdout.flush()
            
            with open(log_path, "w") as f:
                f.write(f"[THREAD] Starting, DATASET_DIR={DATASET_DIR}\n")
                f.write(f"[THREAD] APP_DIR={APP_DIR}\n")
            
            sys.stdout.write(f"[THREAD] Calling train_model_background with {DATASET_DIR}\n")
            sys.stdout.flush()
            
            train_model_background(DATASET_DIR, callback)
            
            sys.stdout.write("[THREAD] Training complete\n")
            sys.stdout.flush()
            write_train_status({"running": False, "progress": 100, "message": "Training complete"})
        except Exception as e:
            error_msg = f"[THREAD] Training error: {str(e)}\n"
            sys.stdout.write(error_msg)
            import traceback
            tb = traceback.format_exc()
            traceback.print_exc()
            sys.stdout.flush()
            
            with open(log_path, "a") as f:
                f.write(error_msg)
                f.write(tb)
            
            write_train_status({"running": False, "progress": 0, "message": str(e)})
    
    sys.stdout.write("Creating thread...\n")
    sys.stdout.flush()
    
    t = threading.Thread(target=train_wrapper, daemon=False)
    t.daemon = False
    t.start()
    
    sys.stdout.write("Thread started\n")
    sys.stdout.flush()
    
    return jsonify({"status":"started"}), 202

# -------- Train progress (polling) --------
@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

# -------- Mark attendance page --------
@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

# -------- Recognize face endpoint (POST image) --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error":"no image"}), 400
    img_file = request.files["image"]
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error":"no face detected"}), 200
        # attempt prediction
        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error":"model not trained"}), 200
        pred_label, conf = predict_with_model(clf, emb)
        # threshold confidence
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200
        # find student name
        student = db["students"].find_one({"_id": ObjectId(pred_label)})
        name = student["name"] if student else "Unknown"
        # save attendance record with timestamp
        ts = datetime.datetime.utcnow().isoformat()
        db["attendance"].insert_one({
            "student_id": pred_label,
            "name": name,
            "timestamp": ts
        })
        return jsonify({"recognized": True, "student_id": pred_label, "name": name, "confidence": float(conf)}), 200
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500

# -------- Attendance records & filters --------
@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")  # all, daily, weekly, monthly
    query = {}
    if period == "daily":
        today = datetime.date.today().isoformat()
        query = {"timestamp": {"$gte": today + "T00:00:00", "$lt": today + "T23:59:59"}}
    elif period == "weekly":
        start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        query = {"timestamp": {"$gte": start}}
    elif period == "monthly":
        start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        query = {"timestamp": {"$gte": start}}
    rows = list(db["attendance"].find(query).sort("timestamp", -1).limit(5000))
    records = [(r.get("_id"), r.get("student_id"), r.get("name"), r.get("timestamp")) for r in rows]
    return render_template("attendance_record.html", records=records, period=period)

# -------- CSV download --------
@app.route("/download_csv", methods=["GET"])
def download_csv():
    rows = list(db["attendance"].find().sort("timestamp", -1))
    output = io.StringIO()
    output.write("id,student_id,name,timestamp\n")
    for r in rows:
        output.write(f'{str(r["_id"])},{r.get("student_id")},{r.get("name")},{r.get("timestamp")}\n')
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

# -------- Students API for listing/editing --------
@app.route("/students", methods=["GET"])
def students_list():
    students = list(db["students"].find().sort("_id", -1))
    data = [ {
        "id": str(s["_id"]),
        "name": s.get("name"),
        "roll": s.get("roll"),
        "class": s.get("class"),
        "section": s.get("section"),
        "reg_no": s.get("reg_no"),
        "created_at": s.get("created_at")
    } for s in students ]
    return jsonify({"students": data})

@app.route("/students/<sid>", methods=["DELETE"])
def delete_student(sid):
    db["students"].delete_one({"_id": ObjectId(sid)})
    db["attendance"].delete_many({"student_id": sid})
    # also delete dataset folder
    folder = os.path.join(DATASET_DIR, sid)
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})

# ---------------- run ------------------------
if __name__ == "__main__":
    app.run(debug=False)