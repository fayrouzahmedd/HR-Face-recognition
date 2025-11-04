# ============================================================
# Face Verification System (Staged Enrollment, NO gTTS)
# - Same visual UI style as app2.6.py
# - Strict gesture-gated enrollment
# - Staged auto-capture (front, right, left)
# - NO TTS
# - Green box shows ONLY the recognized name
# - After logging → "Logged successfully (Attendance/Departure)" banner
# - Uses face_authorized.sqlite3
# ============================================================

import os, sys, cv2, numpy as np, time, threading, csv, shutil, re, traceback, sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import deque
from deepface import DeepFace

# ---------- Optional deps ----------
try:
    import mediapipe as mp
except Exception:
    mp = None
    print("[WARN] MediaPipe not available. Gesture auto-capture disabled.")

# ================ CONFIG ================
FAST_MODE = True

AUTHORIZED_DIR = Path("authorized_faces")
ATTENDANCE_DIR = Path("attendance")
CAM_INDEX      = 0

AUTH_DB_PATH   = Path("face_authorized.sqlite3")
STORE_IMAGES_IN_DB = False   # set True if you want to store crops

# thresholds
ACCEPT_DIST_THRESH = 0.40
REVOKE_DIST_THRESH = 0.50
TOP2_MARGIN_MIN    = 0.08
TOP2_RATIO_MAX     = 0.90

# speed / vision
if FAST_MODE:
    FRAME_DOWNSCALE       = 0.5
    DETECT_EVERY_N_BASE   = 10
    TRACKER_TYPE          = "KCF"
    MODEL_NAME            = "Facenet"
    MAX_TEMPLATES_PER_ID  = 2
    STABLE_FRAMES_AUTH    = 7
    COLOR_HOLD_FRAMES     = 8
    MIN_BBOX_AREA         = 70 * 70
    MIN_LAPLACE_VAR       = 40.0
else:
    FRAME_DOWNSCALE       = 0.7
    DETECT_EVERY_N_BASE   = 6
    TRACKER_TYPE          = "CSRT"
    MODEL_NAME            = "Facenet512"
    MAX_TEMPLATES_PER_ID  = 4
    STABLE_FRAMES_AUTH    = 7
    COLOR_HOLD_FRAMES     = 12
    MIN_BBOX_AREA         = 80 * 80
    MIN_LAPLACE_VAR       = 45.0

# recognition cadence
HEAVY_MIN_PERIOD_SEC  = 0.20
NO_FACE_BACKOFF_MAX_N = 24
NO_FACE_BACKOFF_STEP  = 2

# auto-enroll (gesture)
ENABLE_AUTO_CAPTURE          = True and (mp is not None)
GESTURE_FRAMES_REQUIRED      = 6
REQUIRE_GESTURE_FOR_ENROLL   = True
MIRROR_WEBCAM                = True

# staged capture
STAGE_LIST = [
    ("front",       2, "front"),
    ("right_side",  2, "right"),
    ("left_side",   2, "left"),
]
PRE_CAPTURE_COOLDOWN_SEC = 5.0
STAGE_COOLDOWN_SEC       = 6.0
STAGE_TIMEOUT_PER_STAGE  = 20.0
CAPTURE_IMAGE_INTERVAL   = 0.25

NEW_USER_PREFIX          = "user_"
DIRECT_ENROLL_TO_AUTH    = True
CAPTURE_TRIGGER_COOLDOWN_SEC = 8.0

CAPTURE_ONLY_WHEN_UNAUTHORIZED   = True
CAPTURE_MIN_DIST_FOR_NEW         = 0.55
CAPTURE_SUPPRESS_AFTER_AUTH_SEC  = 15.0

# logging
RELOG_COOLDOWN_SEC = 45.0      # per person
ARRIVAL_HOUR_LIMIT = 18        # <18 => Attendance, else Departure

# UI
DRAW_THICKNESS     = 2
FONT               = cv2.FONT_HERSHEY_SIMPLEX
DIST_SMOOTH_ALPHA  = 0.30
BBOX_SMOOTH_ALPHA  = 0.40

# ================ STATE ================
DB_TEMPLATES: Dict[str, List[np.ndarray]] = {}

_current_identity: Optional[str] = None
_auth_streak = 0
_last_auth_time: Dict[str, float] = {}
_last_auth_seen_ts = 0.0

_tracker = None
_tracker_bbox = None

_last_heavy_submit_ts = 0.0
_detect_every_n = DETECT_EVERY_N_BASE
_no_face_cycles = 0

_last_detect_recog_s = 0.0
_last_auth_latency_s = 0.0
_pending_auth_identity: Optional[str] = None
_pending_auth_start_ts = 0.0

_last_draw_color = (0,0,255)
_last_draw_label = "NOT AUTHORIZED"
_hold_counter = 0
_smoothed_dist: Optional[float] = None
_smoothed_bbox: Optional[Tuple[float,float,float,float]] = None

_capture_in_progress = False
_capture_lock = threading.Lock()
_latest_frame_for_capture = None
_latest_frame_lock = threading.Lock()

_request_db_reload = False
_request_db_reload_lock = threading.Lock()
_pending_staging_queue = deque()
_pending_staging_lock  = threading.Lock()

_last_capture_trigger_ts = 0.0
_last_capture_frame_idx = -999999

# UI overlays
_ui_lock = threading.Lock()
_countdown_active = False
_countdown_end_ts = 0.0
_countdown_label = ""
_flash_capture_until = 0.0
_name_prompt_banner_until = 0.0
_log_banner_text = ""
_log_banner_until = 0.0

# ================ UI helpers ================
def _ui_start_countdown(label: str, seconds: int = 3):
    global _countdown_active, _countdown_end_ts, _countdown_label
    with _ui_lock:
        _countdown_active = True
        _countdown_label  = label
        _countdown_end_ts = time.time() + max(1, seconds) + 0.2

def _ui_stop_countdown():
    global _countdown_active
    with _ui_lock:
        _countdown_active = False

def _ui_flash_capture(duration: float = 0.6):
    global _flash_capture_until
    with _ui_lock:
        _flash_capture_until = time.time() + max(0.1, duration)

def _ui_show_name_prompt_banner(duration: float = 10.0):
    global _name_prompt_banner_until
    with _ui_lock:
        _name_prompt_banner_until = time.time() + max(2.0, duration)

def _ui_show_log_banner(text: str, duration: float = 2.2):
    global _log_banner_text, _log_banner_until
    with _ui_lock:
        _log_banner_text = text
        _log_banner_until = time.time() + duration

# ================ Attendance CSV ================
def _attendance_csv_path() -> Path:
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    return ATTENDANCE_DIR / f"attendance_{day}.csv"

def _ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp_iso", "name", "status", "distance"])

def log_event(name: str, status: str, distance: Optional[float]) -> bool:
    csv_path = _attendance_csv_path()
    _ensure_csv_header(csv_path)
    ts = datetime.now().isoformat(timespec="seconds")
    row = [ts, name, status, "" if distance is None else f"{distance:.4f}"]
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"[LOG] {status} logged for {name}")
        _ui_show_log_banner(f"Logged successfully ({status})", 2.4)
        return True
    except Exception as e:
        print(f"[WARN] Attendance write failed: {e}")
        return False

# ================ DB ================
def _db_connect():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(AUTH_DB_PATH))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_auth_db():
    with _db_connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at REAL NOT NULL
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            emb BLOB NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            stage TEXT,
            img BLOB,
            created_at REAL NOT NULL,
            FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
        )""")

def db_add_person(name: str) -> int:
    with _db_connect() as con:
        con.execute("INSERT OR IGNORE INTO persons(name, created_at) VALUES(?, ?)", (name, time.time()))
        row = con.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
        return int(row[0])

def db_add_template(person_id: int, emb: np.ndarray):
    emb = np.asarray(emb, dtype=np.float32).tobytes()
    with _db_connect() as con:
        con.execute("INSERT INTO templates(person_id, emb, created_at) VALUES(?, ?, ?)",
                    (person_id, sqlite3.Binary(emb), time.time()))

def db_add_image(person_id: int, stage: str, bgr: np.ndarray):
    if not STORE_IMAGES_IN_DB: return
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok: return
    with _db_connect() as con:
        con.execute("INSERT INTO images(person_id, stage, img, created_at) VALUES(?, ?, ?, ?)",
            (person_id, stage, sqlite3.Binary(buf.tobytes()), time.time()))

def db_templates_dict() -> Dict[str, List[np.ndarray]]:
    with _db_connect() as con:
        rows = con.execute("""
            SELECT p.name, t.emb FROM templates t
            JOIN persons p ON p.id = t.person_id
        """).fetchall()
    out: Dict[str, List[np.ndarray]] = {}
    for name, emb_blob in rows:
        e = np.frombuffer(emb_blob, dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-9)
        out.setdefault(name, []).append(e)
    for name, lst in out.items():
        if len(lst) > MAX_TEMPLATES_PER_ID:
            step = max(1, len(lst)//MAX_TEMPLATES_PER_ID)
            out[name] = lst[::step][:MAX_TEMPLATES_PER_ID]
    return out

# ================ Detection / Embedding ================
FRONTAL_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def _dynamic_min_size(w: int, h: int) -> Tuple[int, int]:
    m = max(32, int(min(w, h) * 0.15))
    m = min(m, max(32, min(w, h)))
    return (m, m)

def _detect_with_cascade(gray, cascade, min_size):
    try:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=min_size
        )
        return faces
    except Exception:
        return ()

def _bbox_from_faces(faces):
    if len(faces) == 0: return None
    x, y, ww, hh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    return (x, y, x+ww, y+hh)

def detect_largest_face_bbox(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if bgr is None or bgr.size == 0: return None
    h, w = bgr.shape[:2]
    if w < 32 or h < 32: return None
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    mn = _dynamic_min_size(w, h)

    faces = _detect_with_cascade(gray, FRONTAL_CASCADE, mn)
    if len(faces):
        return _bbox_from_faces(faces)

    faces = _detect_with_cascade(gray, PROFILE_CASCADE, mn)
    if len(faces):
        return _bbox_from_faces(faces)

    gray_flipped = cv2.flip(gray, 1)
    faces = _detect_with_cascade(gray_flipped, PROFILE_CASCADE, mn)
    if len(faces):
        x, y, ww, hh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        x1 = w - (x + ww); y1 = y; x2 = w - x; y2 = y + hh
        return (x1, y1, x2, y2)

    return None

def detect_face_for_stage(bgr: np.ndarray, stage_name: str) -> Optional[Tuple[int,int,int,int]]:
    return detect_largest_face_bbox(bgr)

def is_crop_usable(bgr_crop: np.ndarray) -> bool:
    if bgr_crop is None or bgr_crop.size == 0:
        return False
    h, w = bgr_crop.shape[:2]
    if (w * h) < MIN_BBOX_AREA:
        return False
    g = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(g, cv2.CV_64F).var() < MIN_LAPLACE_VAR:
        return False
    return True

def build_model():
    try:
        return (DeepFace.build_model(MODEL_NAME), None)
    except Exception:
        return (DeepFace.build_model("Facenet512"), None)

def embed_bgr_crop(model_pair, bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        reps = DeepFace.represent(img_path=rgb, model_name=MODEL_NAME,
                                  detector_backend="skip", enforce_detection=False)
        if not reps:
            return None
        e = np.array(reps[0]["embedding"], dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-9)
        return e
    except Exception as e:
        print("[WARN] embedding failed:", e)
        return None

# ================ Migration from authorized_faces/ ================
def load_authorized_db_from_folder(model_pair, directory: Path) -> Dict[str, List[np.ndarray]]:
    if not directory.exists():
        return {}
    db = {}
    for sub in directory.glob("*"):
        if not sub.is_dir():
            continue
        name = sub.name
        embs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            for p in sub.glob(ext):
                img = cv2.imread(str(p))
                if img is None: continue
                bbox = detect_largest_face_bbox(img)
                if bbox is None: continue
                x1,y1,x2,y2 = bbox
                crop = img[y1:y2, x1:x2]
                if not is_crop_usable(crop): continue
                e = embed_bgr_crop(model_pair, crop)
                if e is not None:
                    embs.append(e)
        if embs:
            if len(embs) > MAX_TEMPLATES_PER_ID:
                embs = embs[:MAX_TEMPLATES_PER_ID]
            db[name] = embs
    return db

def migrate_folder_to_sqlite_if_needed(model_pair):
    with _db_connect() as con:
        n = con.execute("SELECT COUNT(1) FROM persons").fetchone()[0]
    if n > 0:
        return
    if not AUTHORIZED_DIR.exists():
        return
    print("[MIGRATE] importing from authorized_faces/ ...")
    folder_db = load_authorized_db_from_folder(model_pair, AUTHORIZED_DIR)
    for name, embs in folder_db.items():
        pid = db_add_person(name)
        for e in embs:
            db_add_template(pid, e)
    print("[MIGRATE] done")

# ================ Matching ================
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))

def _match(db, emb):
    pairs = []
    for name, refs in db.items():
        if not refs:
            continue
        dmin = min(cosine_distance(emb, r) for r in refs)
        pairs.append((name, dmin))
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[1])
    best_name, best_dist = pairs[0]
    second = pairs[1][1] if len(pairs) > 1 else 1.0
    print(f"[MATCH] best={best_name} d={best_dist:.4f} second={second:.4f}")
    return best_name, best_dist, second

def recognize_heavy(model_pair, db, bgr):
    t0 = time.perf_counter()
    bbox = detect_largest_face_bbox(bgr)
    if bbox is None:
        return None
    x1,y1,x2,y2 = bbox
    crop = bgr[y1:y2, x1:x2]
    if not is_crop_usable(crop):
        return None
    emb = embed_bgr_crop(model_pair, crop)
    if emb is None:
        return None
    m = _match(db, emb)
    if m is None:
        return None
    best_name, best_dist, second = m
    clear_margin = (second - best_dist) >= TOP2_MARGIN_MIN
    clear_ratio  = (best_dist / max(second, 1e-9)) <= TOP2_RATIO_MAX
    accept = best_dist <= ACCEPT_DIST_THRESH
    is_auth_fast = accept and clear_margin and clear_ratio
    return (bbox, best_name, best_dist, is_auth_fast, time.perf_counter() - t0)

# ================ Gesture helpers ================
def _landmark_to_pixel(lm, img_w, img_h):
    return (int(round(lm.x*img_w)), int(round(lm.y*img_h)))

def _finger_states(landmarks, handedness_label, img_w, img_h):
    states = {"thumb":False,"index":False,"middle":False,"ring":False,"pinky":False}
    tip = {"thumb":4,"index":8,"middle":12,"ring":16,"pinky":20}
    pip = {"thumb":2,"index":6,"middle":10,"ring":14,"pinky":18}
    try:
        pts = {i:_landmark_to_pixel(l, img_w, img_h) for i,l in enumerate(landmarks)}
        for f in ("index","middle","ring","pinky"):
            states[f] = pts[tip[f]][1] < pts[pip[f]][1]
        handed_right = True if handedness_label is None else handedness_label.lower()=="right"
        if MIRROR_WEBCAM:
            handed_right = not handed_right
        states["thumb"] = (pts[tip["thumb"]][0] > pts[pip["thumb"]][0]+6) if handed_right else (pts[tip["thumb"]][0] < pts[pip["thumb"]][0]-6)
    except Exception:
        pass
    return states

def _three_up(mp_results, w, h):
    if mp_results is None or not getattr(mp_results, "multi_hand_landmarks", None):
        return False, None
    hand = mp_results.multi_hand_landmarks[0]
    label = None
    try:
        if getattr(mp_results, "multi_handedness", None):
            label = mp_results.multi_handedness[0].classification[0].label
    except Exception:
        pass
    states = _finger_states(hand.landmark, label, w, h)
    ok = states["index"] and states["middle"] and states["ring"]
    return ok, (states, label)

# ================ Capture Worker ================
class CaptureWorker:
    def __init__(self): self._thread=None
    def start_capture(self):
        if self._thread is not None and self._thread.is_alive(): return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True
    def _run(self):
        global _capture_in_progress
        try:
            with _capture_lock:
                if _capture_in_progress: return
                _capture_in_progress = True

            base_dir = AUTHORIZED_DIR if DIRECT_ENROLL_TO_AUTH else Path("staging_enroll")
            base_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_name = ("_pending_" if DIRECT_ENROLL_TO_AUTH else "") + NEW_USER_PREFIX + ts
            target = base_dir / ts_name
            i = 0
            while target.exists():
                i += 1
                target = base_dir / f"{ts_name}_{i}"
            target.mkdir(parents=True, exist_ok=True)
            print(f"[CAPTURE] saving into: {target}")

            # initial cooldown
            time.sleep(PRE_CAPTURE_COOLDOWN_SEC)

            total_saved = 0
            for idx, (stage_name, stage_count, _) in enumerate(STAGE_LIST, start=1):
                _ui_start_countdown(f"{stage_name} in", 3)
                time.sleep(3.2)
                _ui_stop_countdown()

                saved = 0
                last_save = 0.0
                deadline = time.time() + STAGE_TIMEOUT_PER_STAGE
                while saved < stage_count and time.time() < deadline:
                    with _latest_frame_lock:
                        f = None if _latest_frame_for_capture is None else _latest_frame_for_capture.copy()
                    if f is None or f.size == 0:
                        time.sleep(0.05); continue
                    bbox = detect_face_for_stage(f, stage_name)
                    if bbox is None:
                        time.sleep(0.04); continue
                    x1,y1,x2,y2 = bbox
                    h,w = f.shape[:2]
                    x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(w-1,x2),min(h-1,y2)
                    if x2 <= x1 or y2 <= y1:
                        time.sleep(0.04); continue
                    now = time.time()
                    if (now - last_save) < CAPTURE_IMAGE_INTERVAL:
                        time.sleep(0.03); continue
                    pad = 0.12
                    dx = int((x2-x1)*pad); dy = int((y2-y1)*pad)
                    x1p=max(0,x1-dx); y1p=max(0,y1-dy); x2p=min(w-1,x2+dx); y2p=min(h-1,y2+dy)
                    crop = f[y1p:y2p, x1p:x2p]
                    if crop is None or crop.size == 0:
                        time.sleep(0.03); continue
                    out = target / f"{target.name}_{stage_name}_{saved+1:02d}.jpg"
                    _ui_flash_capture(0.6)
                    ok = cv2.imwrite(str(out), crop)
                    print(f"[CAPTURE] [{stage_name}] saved {out.name} ok={ok}")
                    if ok:
                        saved += 1
                        total_saved += 1
                        last_save = now

                if idx < len(STAGE_LIST):
                    time.sleep(STAGE_COOLDOWN_SEC)

            if total_saved == 0:
                try: shutil.rmtree(target)
                except Exception: pass
                print("[CAPTURE] nothing saved")
                return

            with _pending_staging_lock:
                _pending_staging_queue.append(target)
            print(f"[CAPTURE] ready: {target} (saved={total_saved})")

        except Exception as e:
            print("[CAPTURE] error:", e)
            traceback.print_exc()
        finally:
            with _capture_lock:
                _capture_in_progress = False

# ================ Enrollment finalize ================
def embed_from_folder(model_pair, folder: Path) -> Optional[np.ndarray]:
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        for p in sorted(folder.glob(ext)):
            img = cv2.imread(str(p))
            if img is None: continue
            bbox = detect_largest_face_bbox(img)
            if bbox is None: continue
            x1,y1,x2,y2 = bbox
            crop = img[y1:y2, x1:x2]
            if not is_crop_usable(crop): continue
            e = embed_bgr_crop(model_pair, crop)
            if e is not None:
                return e
    return None

def duplicate_check(emb) -> Optional[Tuple[str, float]]:
    best = None; best_d = 1e9
    for name, refs in DB_TEMPLATES.items():
        if not refs: continue
        d = min(cosine_distance(emb, r) for r in refs)
        if d < best_d:
            best_d = d; best = name
    return (best, best_d) if best is not None else None

def _timed_input(prompt: str, timeout: float, default: str) -> str:
    sys.stdout.write(prompt); sys.stdout.flush()
    result = {"text": default}
    def _reader():
        try:
            txt = input().strip()
            if txt: result["text"] = txt
        except Exception:
            pass
    if not sys.stdin or not sys.stdin.isatty():
        print("")
        return default
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print("")
    return result["text"]

def prompt_name(default_name: str) -> str:
    _ui_show_name_prompt_banner(10.0)
    entered = _timed_input(f"New user name (default '{default_name}', 10s): ", 10.0, default_name)
    entered = re.sub(r"\s+", "_", entered)
    entered = re.sub(r"[^A-Za-z0-9_\-]", "", entered)[:64]
    return entered or default_name

def finalize_one(model_pair, folder_path: Path):
    global _request_db_reload
    if folder_path is None or not folder_path.exists():
        return
    print(f"[ENROLL] finalizing {folder_path}")
    emb = embed_from_folder(model_pair, folder_path)
    if emb is None:
        print("[ENROLL] no usable embedding; removing")
        try: shutil.rmtree(folder_path)
        except Exception: pass
        return
    dup = duplicate_check(emb)
    if dup is not None:
        name, d = dup
        print(f"[ENROLL] duplicate '{name}' d={d:.3f}")
        if d <= ACCEPT_DIST_THRESH:
            try: shutil.rmtree(folder_path)
            except Exception: pass
            return
    default_name = folder_path.name.replace("_pending_", "")
    final_name = prompt_name(default_name)
    try:
        pid = db_add_person(final_name)
        embs_added = 0
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            for p in sorted(folder_path.glob(ext)):
                img = cv2.imread(str(p))
                if img is None: continue
                bbox = detect_largest_face_bbox(img)
                if bbox is None: continue
                x1,y1,x2,y2 = bbox
                crop = img[y1:y2, x1:x2]
                if not is_crop_usable(crop): continue
                e = embed_bgr_crop(model_pair, crop)
                if e is None: continue
                db_add_template(pid, e); embs_added += 1
                if STORE_IMAGES_IN_DB:
                    db_add_image(pid, "auto", crop)
                if embs_added >= MAX_TEMPLATES_PER_ID:
                    break
            if embs_added >= MAX_TEMPLATES_PER_ID:
                break
        print(f"[ENROLL] done for '{final_name}' (templates: {embs_added})")
    except Exception as e:
        print("[ENROLL] SQLite finalize failed:", e)
    try: shutil.rmtree(folder_path)
    except Exception: pass
    with _request_db_reload_lock:
        _request_db_reload = True

def finalize_all(model_pair):
    while True:
        with _pending_staging_lock:
            if not _pending_staging_queue:
                return
            folder = _pending_staging_queue.popleft()
        finalize_one(model_pair, folder)

# ------------------- ENROLLMENT GATING -------------------
def enroll_allowed_now() -> Tuple[bool, str]:
    if _capture_in_progress:
        return False, "capture already in progress"
    now_wall = time.time()
    if (now_wall - _last_capture_trigger_ts) < CAPTURE_TRIGGER_COOLDOWN_SEC:
        return False, "global capture cooldown"
    if _current_identity is not None:
        return False, "face already exists in DB (named match)"
    if _smoothed_dist is not None and _smoothed_dist <= ACCEPT_DIST_THRESH:
        return False, "face already exists in DB (distance <= accept)"
    if CAPTURE_ONLY_WHEN_UNAUTHORIZED:
        if (now_wall - _last_auth_seen_ts) < CAPTURE_SUPPRESS_AFTER_AUTH_SEC:
            return False, "recent authorization suppression"
        if _smoothed_dist is not None and _smoothed_dist < CAPTURE_MIN_DIST_FOR_NEW:
            return False, "face too similar to existing identities"
    return True, "ok"

# ================ Main =================
def main():
    global DB_TEMPLATES
    global _last_heavy_submit_ts, _detect_every_n, _no_face_cycles
    global _current_identity, _auth_streak, _smoothed_dist, _smoothed_bbox
    global _last_detect_recog_s, _last_auth_latency_s, _pending_auth_identity, _pending_auth_start_ts
    global _latest_frame_for_capture, _last_auth_seen_ts, _last_capture_trigger_ts, _last_capture_frame_idx
    global _tracker, _tracker_bbox, _hold_counter, _last_draw_color, _last_draw_label
    global _request_db_reload
    # 👇 these were missing and caused UnboundLocalError
    global _countdown_active, _countdown_end_ts, _countdown_label
    global _flash_capture_until, _name_prompt_banner_until
    global _log_banner_text, _log_banner_until

    model_pair = build_model()
    init_auth_db()
    migrate_folder_to_sqlite_if_needed(model_pair)
    DB_TEMPLATES = db_templates_dict()
    AUTHORIZED_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] camera open failed")
        return

    class Recognizer:
        def __init__(self, model_pair):
            self.model = model_pair
            self.lock = threading.Lock()
            self.busy = False
            self.frame = None
            self.res = None
            self.stop = False
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        def submit(self, frame):
            if self.busy:
                return False
            with self.lock:
                self.frame = frame.copy()
                self.busy = True
            return True
        def get(self):
            return self.res
        def _loop(self):
            while not self.stop:
                f = None
                with self.lock:
                    if self.frame is not None:
                        f = self.frame
                        self.frame = None
                if f is None:
                    time.sleep(0.01)
                    continue
                try:
                    r = recognize_heavy(self.model, DB_TEMPLATES, f)
                    self.res = r
                except Exception as e:
                    print("[WARN] recognizer failure:", e)
                    self.res = None
                finally:
                    with self.lock:
                        self.busy = False
        def shutdown(self):
            self.stop = True
            try:
                self.thread.join(timeout=0.5)
            except Exception:
                pass

    recog = Recognizer(model_pair)
    capt  = CaptureWorker()

    mp_hands = None
    if ENABLE_AUTO_CAPTURE and mp is not None:
        try:
            mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55
            )
            print("[AUTO] MediaPipe Hands ON")
        except Exception as e:
            print("[AUTO] MediaPipe failed:", e)
            mp_hands = None

    print("[INFO] Running. Press 'q' to quit.")
    loop_ts_local = time.perf_counter()
    frame_idx_local = 0
    gesture_count_local = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] camera read fail")
                break
            if MIRROR_WEBCAM:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - loop_ts_local
            loop_ts_local = now
            fps = (1.0 / dt) if dt > 0 else 0.0

            frame_disp = cv2.resize(frame, None, fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE) if FRAME_DOWNSCALE != 1.0 else frame
            with _latest_frame_lock:
                _latest_frame_for_capture = frame.copy()
            frame_idx_local += 1

            # --- gesture ---
            if ENABLE_AUTO_CAPTURE and mp_hands is not None and not _capture_in_progress:
                try:
                    res = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                except Exception:
                    res = None
                three, _ = _three_up(res, frame.shape[1], frame.shape[0])
                if three:
                    gesture_count_local += 1
                else:
                    gesture_count_local = max(0, gesture_count_local - 1)
                gesture_count_local = max(0, min(gesture_count_local, GESTURE_FRAMES_REQUIRED))

                # bottom bar
                bar_w, bar_h, x0, y0 = 140, 18, 10, frame_disp.shape[0] - 30
                progress = int((gesture_count_local / max(1, GESTURE_FRAMES_REQUIRED)) * bar_w)
                cv2.rectangle(frame_disp, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
                cv2.rectangle(frame_disp, (x0, y0), (x0 + progress, y0 + bar_h), (0, 200, 0), -1)
                cv2.putText(
                    frame_disp,
                    f"Gesture progress: {min(gesture_count_local, GESTURE_FRAMES_REQUIRED)}/{GESTURE_FRAMES_REQUIRED}",
                    (x0 + bar_w + 8, y0 + bar_h - 3), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

                if gesture_count_local >= GESTURE_FRAMES_REQUIRED and _last_capture_frame_idx != frame_idx_local:
                    allowed, reason = enroll_allowed_now()
                    if REQUIRE_GESTURE_FOR_ENROLL and allowed:
                        now_wall = time.time()
                        if capt.start_capture():
                            _last_capture_trigger_ts = now_wall
                            _last_capture_frame_idx = frame_idx_local
                            print("[AUTO] capture started")
                        else:
                            print("[AUTO] worker busy")
                    else:
                        print(f"[AUTO] capture blocked: {reason}")
                    gesture_count_local = 0

            # --- Recognition cadence ---
            run_heavy_due_to_cadence = (frame_idx_local % _detect_every_n) == 0 or (_tracker is None)
            run_heavy_due_to_period  = (time.time() - _last_heavy_submit_ts) >= HEAVY_MIN_PERIOD_SEC
            if run_heavy_due_to_cadence and run_heavy_due_to_period:
                _last_heavy_submit_ts = time.time()
                recog.submit(frame_disp)

                rec = recog.get()
                if rec is None:
                    _hold_counter = max(0, _hold_counter - 1)
                    _no_face_cycles = min(_no_face_cycles + 1, 50)
                    _detect_every_n = min(DETECT_EVERY_N_BASE + _no_face_cycles * NO_FACE_BACKOFF_STEP,
                                          NO_FACE_BACKOFF_MAX_N)
                else:
                    _no_face_cycles = 0
                    _detect_every_n = DETECT_EVERY_N_BASE

                    (bbox, identity, dist_raw, is_auth_raw, secs) = rec
                    _last_detect_recog_s = secs

                    # smooth distance
                    _smoothed_dist = dist_raw if _smoothed_dist is None else (DIST_SMOOTH_ALPHA * dist_raw + (1 - DIST_SMOOTH_ALPHA) * _smoothed_dist)

                    # smooth bbox
                    x1, y1, x2, y2 = bbox
                    if _smoothed_bbox is None:
                        _smoothed_bbox = (x1, y1, x2, y2)
                    else:
                        sx1, sy1, sx2, sy2 = _smoothed_bbox
                        _smoothed_bbox = (
                            BBOX_SMOOTH_ALPHA * x1 + (1 - BBOX_SMOOTH_ALPHA) * sx1,
                            BBOX_SMOOTH_ALPHA * y1 + (1 - BBOX_SMOOTH_ALPHA) * sy1,
                            BBOX_SMOOTH_ALPHA * x2 + (1 - BBOX_SMOOTH_ALPHA) * sx2,
                            BBOX_SMOOTH_ALPHA * y2 + (1 - BBOX_SMOOTH_ALPHA) * sy2,
                        )
                    x1, y1, x2, y2 = map(lambda v: int(round(v)), _smoothed_bbox)

                    # authorized?
                    if _current_identity == identity and _current_identity is not None:
                        is_authorized = (_smoothed_dist is not None and _smoothed_dist <= REVOKE_DIST_THRESH)
                    else:
                        is_authorized = (_smoothed_dist is not None and _smoothed_dist <= ACCEPT_DIST_THRESH)

                    if is_authorized and identity != "Unknown":
                        if identity == _current_identity:
                            _auth_streak += 1
                        else:
                            _current_identity = identity
                            _auth_streak = 1
                            _pending_auth_identity = identity
                            _pending_auth_start_ts = time.perf_counter()
                    else:
                        _current_identity = None
                        _auth_streak = 0
                        _pending_auth_identity = None

                    # UI label: ONLY the name if authorized
                    if is_authorized and identity != "Unknown":
                        desired_label = f"{identity}"
                        desired_color = (0, 200, 0)
                    else:
                        desired_label = "NOT AUTHORIZED"
                        desired_color = (0, 0, 255)

                    if (desired_color != _last_draw_color) or (desired_label != _last_draw_label):
                        if _hold_counter <= 0:
                            _last_draw_color = desired_color
                            _last_draw_label = desired_label
                            _hold_counter = COLOR_HOLD_FRAMES
                    else:
                        _hold_counter = max(0, _hold_counter - 1)

                    # draw box
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), _last_draw_color, DRAW_THICKNESS)
                    (tw, th), _ = cv2.getTextSize(_last_draw_label, FONT, 0.6, 2)
                    cv2.rectangle(frame_disp, (x1, y2 + 4), (x1 + tw + 6, y2 + th + 10), _last_draw_color, -1)
                    cv2.putText(frame_disp, _last_draw_label, (x1 + 3, y2 + th + 3), FONT, 0.6, (255, 255, 255), 2)

                    # attendance/departure logging
                    if _auth_streak >= STABLE_FRAMES_AUTH and _current_identity is not None:
                        t_fire = time.perf_counter()
                        if _pending_auth_identity == _current_identity and _pending_auth_start_ts > 0:
                            _last_auth_latency_s = (t_fire - _pending_auth_start_ts)
                        now_wall = time.time()
                        last_t = _last_auth_time.get(_current_identity, 0.0)
                        if (now_wall - last_t) >= RELOG_COOLDOWN_SEC:
                            now_local = datetime.now()
                            if now_local.hour < ARRIVAL_HOUR_LIMIT:
                                status = "Attendance"
                            else:
                                status = "Departure"
                            log_event(_current_identity, status, _smoothed_dist)
                            _last_auth_time[_current_identity] = now_wall
                        _last_auth_seen_ts = time.time()
                        _auth_streak = 0
                        _pending_auth_identity = None

                    # tracker init
                    try:
                        if TRACKER_TYPE.upper() == "KCF":
                            try:
                                _tracker = cv2.legacy.TrackerKCF_create()
                            except Exception:
                                _tracker = cv2.TrackerKCF_create()
                        else:
                            try:
                                _tracker = cv2.legacy.TrackerCSRT_create()
                            except Exception:
                                _tracker = cv2.TrackerCSRT_create()
                        _tracker_bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                        _tracker.init(frame_disp, _tracker_bbox)
                    except Exception:
                        _tracker = None
                        _tracker_bbox = None
            else:
                # tracker-only draw
                if _tracker is not None and _tracker_bbox is not None:
                    ok_tr, tbox = _tracker.update(frame_disp)
                    if ok_tr:
                        x, y, w, h = map(int, tbox)
                        if _smoothed_bbox is None:
                            _smoothed_bbox = (x, y, x + w, y + h)
                        else:
                            sx1, sy1, sx2, sy2 = _smoothed_bbox
                            _smoothed_bbox = (
                                BBOX_SMOOTH_ALPHA * x + (1 - BBOX_SMOOTH_ALPHA) * sx1,
                                BBOX_SMOOTH_ALPHA * y + (1 - BBOX_SMOOTH_ALPHA) * sy1,
                                BBOX_SMOOTH_ALPHA * (x + w) + (1 - BBOX_SMOOTH_ALPHA) * sx2,
                                BBOX_SMOOTH_ALPHA * (y + h) + (1 - BBOX_SMOOTH_ALPHA) * sy2,
                            )
                        sx1, sy1, sx2, sy2 = map(lambda v: int(round(v)), _smoothed_bbox)
                        cv2.rectangle(frame_disp, (sx1, sy1), (sx2, sy2), _last_draw_color, DRAW_THICKNESS)
                        (tw, th), _ = cv2.getTextSize(_last_draw_label, FONT, 0.6, 2)
                        cv2.rectangle(frame_disp, (sx1, sy2 + 4), (sx1 + tw + 6, sy2 + th + 10), _last_draw_color, -1)
                        cv2.putText(frame_disp, _last_draw_label, (sx1 + 3, sy2 + th + 3), FONT, 0.6,
                                    (255, 255, 255), 2)
                        _hold_counter = max(0, _hold_counter - 1)
                    else:
                        _tracker = None
                        _tracker_bbox = None

            # finalize enrollments, reload DB
            finalize_all(model_pair)
            with _request_db_reload_lock:
                if _request_db_reload:
                    try:
                        DB_TEMPLATES = db_templates_dict()
                        print("[DB] reloaded")
                    except Exception as e:
                        print("[DB] reload failed:", e)
                    _request_db_reload = False

            # --- draw countdown / flash / name prompt / log banner ---
            now_wall = time.time()
            with _ui_lock:
                if _countdown_active:
                    secs_left = int(max(0, round(_countdown_end_ts - now_wall)))
                    if secs_left > 0:
                        overlay = frame_disp.copy()
                        cv2.rectangle(overlay, (0,0), (overlay.shape[1], overlay.shape[0]), (0,0,0), -1)
                        frame_disp = cv2.addWeighted(overlay, 0.35, frame_disp, 0.65, 0)
                        label = f"{_countdown_label}"
                        (tw,th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
                        cx = frame_disp.shape[1]//2 - tw//2
                        cy = frame_disp.shape[0]//2 - 80
                        cv2.putText(frame_disp, label, (cx, cy), FONT, 0.9, (255,255,255), 2, cv2.LINE_AA)
                        num = str(secs_left)
                        (tw2,th2), _ = cv2.getTextSize(num, FONT, 3.0, 8)
                        cx2 = frame_disp.shape[1]//2 - tw2//2
                        cy2 = frame_disp.shape[0]//2 + th2//2
                        cv2.putText(frame_disp, num, (cx2, cy2), FONT, 3.0, (0,255,0), 8, cv2.LINE_AA)
                    else:
                        _countdown_active = False

                if _flash_capture_until > now_wall:
                    cv2.rectangle(frame_disp, (4,4), (frame_disp.shape[1]-4, frame_disp.shape[0]-4), (0,255,0), 6)
                    text = "Capturing..."
                    (tw,th), _ = cv2.getTextSize(text, FONT, 0.9, 3)
                    x = frame_disp.shape[1]//2 - tw//2
                    cv2.rectangle(frame_disp, (x-10, 45-th-10), (x+tw+10, 45+10), (0,255,0), -1)
                    cv2.putText(frame_disp, text, (x, 45), FONT, 0.9, (0,0,0), 3, cv2.LINE_AA)

                if _name_prompt_banner_until > now_wall:
                    msg = "Please write your name in console and press Enter..."
                    (tw,th), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
                    cv2.rectangle(frame_disp, (10,10), (10+tw+14, 10+th+20), (60,60,220), -1)
                    cv2.putText(frame_disp, msg, (17, 10+th+10), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)

                if _log_banner_until > now_wall and _log_banner_text:
                    (tw, th), _ = cv2.getTextSize(_log_banner_text, FONT, 0.7, 2)
                    x1 = frame_disp.shape[1] - tw - 20
                    y1 = 16
                    cv2.rectangle(frame_disp, (x1 - 6, y1 - th - 10), (x1 + tw + 6, y1 + 10), (0,150,0), -1)
                    cv2.putText(frame_disp, _log_banner_text, (x1, y1), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # small stats
            y = 24
            def put(t):
                nonlocal y
                cv2.putText(frame_disp, t, (10, y), FONT, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame_disp, t, (10, y), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)
                y += 22
            put(f"Detect+Recognize: {_last_detect_recog_s:.3f} s")
            put(f"Auth latency (last): {_last_auth_latency_s:.3f} s")
            put(f"FPS: {fps:.1f}")
            put(f"Heavy cadence: every ~{_detect_every_n} frames")

            cv2.imshow("Face Verification System (Staged Enrollment, NO gTTS)", frame_disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        try:
            recog.shutdown()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass
    main()