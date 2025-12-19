import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

def crop_face_and_embed(bgr_image, detection=None):
    h, w = bgr_image.shape[:2]
    if detection is not None:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(max(0, bbox.xmin * w))
        y1 = int(max(0, bbox.ymin * h))
        x2 = int(min(w, (bbox.xmin + bbox.width) * w))
        y2 = int(min(h, (bbox.ymin + bbox.height) * h))
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    if x2 <= x1 or y2 <= y1:
        return None
    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32,32), interpolation=cv2.INTER_AREA)
    emb = face.flatten().astype(np.float32) / 255.0
    return emb

def extract_embedding_for_image(stream_or_bytes):
    if not MEDIAPIPE_AVAILABLE:
        data = stream_or_bytes.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return crop_face_and_embed(img, detection=None)
    
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.1)
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return None
    emb = crop_face_and_embed(img, results.detections[0])
    return emb

def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_with_model(clf, emb):
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    return label, conf

def train_model_background(dataset_dir, progress_callback=None):
    import sys
    import os
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_debug.log")
    
    sys.stdout.write(f"[TRAIN] Starting training with dataset_dir: {dataset_dir}\n")
    sys.stdout.flush()
    
    with open(log_path, "w") as log:
        log.write(f"[TRAIN] Starting training with dataset_dir: {dataset_dir}\n")
        log.write(f"[TRAIN] Log path: {log_path}\n")
    
    try:
        X = []
        y = []
        
        with open(log_path, "a") as log:
            log.write(f"[TRAIN] Listing directories: {dataset_dir}\n")
        
        sys.stdout.write(f"[TRAIN] Listing directories...\n")
        sys.stdout.flush()
        
        student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        with open(log_path, "a") as log:
            log.write(f"[TRAIN] Found {len(student_dirs)} student folders: {student_dirs}\n")
        
        sys.stdout.write(f"[TRAIN] Found {len(student_dirs)} student folders\n")
        sys.stdout.flush()
        
        if not student_dirs:
            msg = "No student folders found"
            sys.stdout.write(f"[TRAIN] {msg}\n")
            sys.stdout.flush()
            if progress_callback:
                progress_callback(0, msg)
            return
        
        total_students = len(student_dirs)
        processed = 0

        for sid in student_dirs:
            folder = os.path.join(dataset_dir, sid)
            sys.stdout.write(f"[TRAIN] Processing student {sid}\n")
            sys.stdout.flush()
            
            try:
                files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
                
                with open(log_path, "a") as log:
                    log.write(f"[TRAIN] Found {len(files)} images for student {sid}\n")
                
                sys.stdout.write(f"[TRAIN] Found {len(files)} images for student {sid}\n")
                sys.stdout.flush()
                
                for fn in files:
                    path = os.path.join(folder, fn)
                    sys.stdout.write(f"[TRAIN] Reading {path}\n")
                    sys.stdout.flush()
                    img = cv2.imread(path)
                    if img is None:
                        with open(log_path, "a") as log:
                            log.write(f"[TRAIN] Failed to read {path}\n")
                        sys.stdout.write(f"[TRAIN] Failed to read {path}\n")
                        sys.stdout.flush()
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
                    emb = gray.flatten().astype(np.float32) / 255.0
                    X.append(emb)
                    y.append(sid)
                    sys.stdout.write(f"[TRAIN] Added image for {sid}, total samples: {len(X)}\n")
                    sys.stdout.flush()
            except Exception as e:
                error_msg = f"[TRAIN] ERROR processing student {sid}: {str(e)}\n"
                sys.stdout.write(error_msg)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                
                with open(log_path, "a") as log:
                    log.write(error_msg)
                    log.write(traceback.format_exc())
            
            processed += 1
            msg = f"Processed {processed}/{total_students} students"
            sys.stdout.write(f"[TRAIN] {msg}\n")
            sys.stdout.flush()
            if progress_callback:
                pct = int((processed/total_students)*80)
                progress_callback(pct, msg)

        if len(X) == 0:
            msg = "No training data found"
            sys.stdout.write(f"[TRAIN] {msg}\n")
            sys.stdout.flush()
            if progress_callback:
                progress_callback(0, msg)
            return

        with open(log_path, "a") as log:
            log.write(f"[TRAIN] Converting to numpy arrays, X size: {len(X)}, y size: {len(y)}\n")
        
        X = np.array(X)
        y = np.array(y)

        with open(log_path, "a") as log:
            log.write(f"[TRAIN] Training RandomForest with {len(X)} samples\n")

        sys.stdout.write(f"[TRAIN] Training RandomForest with {len(X)} samples\n")
        sys.stdout.flush()
        
        if progress_callback:
            progress_callback(85, "Training RandomForest...")
        
        clf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
        clf.fit(X, y)
        
        with open(log_path, "a") as log:
            log.write(f"[TRAIN] Model trained successfully\n")

        sys.stdout.write(f"[TRAIN] Saving model to {MODEL_PATH}\n")
        sys.stdout.flush()
        
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        
        with open(log_path, "a") as log:
            log.write(f"[TRAIN] Model saved to {MODEL_PATH}\n")

        sys.stdout.write(f"[TRAIN] Training complete!\n")
        sys.stdout.flush()
        if progress_callback:
            progress_callback(100, "Training complete")
            
    except Exception as e:
        error_msg = f"[TRAIN] ERROR: {str(e)}\n"
        sys.stdout.write(error_msg)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        with open(log_path, "a") as log:
            log.write(error_msg)
            log.write(traceback.format_exc())
        
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
