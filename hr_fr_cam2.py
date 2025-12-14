# Face recognition attendance (SQLite) — FINAL with:
# - Multi-face detection & recognition
# - No MediaPipe / hand gesture logic
# - Enrollment is triggered ONLY by pressing 'g'
# - Minimal UI (name label; brief banner after logging)
# - Attendance/Departure events are stored in SQLite (table: attendance), not CSV
# - Arabic TTS added (gTTS, cached, non-blocking)
# - New users logged into a separate SQLite table (new_users_log) upon enrollment
# - Arabic welcome/goodbye for authorized users on Arrival/Departure

import os, sys, cv2, numpy as np, time, threading, shutil, re, traceback, pickle, sqlite3, hashlib, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import deque

# -------- DeepFace (soft) --------
try:
    from deepface import DeepFace
except Exception as e:
    DeepFace = None
    print("[WARN] DeepFace is not available in this environment:", e)

# ================ CONFIG ================
FAST_MODE = True

AUTHORIZED_DIR = Path("authorized_faces")
CACHE_PATH     = Path("attendance") / "emb_cache.pkl"  # kept for legacy cache location
CACHE_ENABLED  = True


# ---- Camera config ----
USE_IP_CAMERA = True  # True → use Hikvision IP cam, False → use local webcam
CAM_INDEX     = 0     # used only if USE_IP_CAMERA = False

# Hikvision RTSP URL:
# main stream: /Streaming/Channels/101
# sub stream:  /Streaming/Channels/102  (lower resolution, usually better for realtime)
#
# IMPORTANT:
# - Replace YOUR_PASSWORD with the real one.
# - If the password contains '@', ':' or spaces, URL-encode them (e.g. '@' → '%40')
IP_CAM_URL = "rtsp://admin:Mori%40111288@192.168.1.64:554/Streaming/Channels/102"



# ----------------------------------------------------------- #
## stream_url = "url ip cam (full vlc media player)"
## stream_url= "rtsp://admin:Mori@111288@192.168.1.64:554"
# ----------------------------------------------------------- #


AUTH_DB_PATH = Path("face_authorized.sqlite3")
STORE_IMAGES_IN_DB = False

# recognition thresholds
# recognition thresholds  (STRONGER / SAFER)
# recognition thresholds  (BALANCED)
ACCEPT_DIST_THRESH = 0.38      # was 0.33
REVOKE_DIST_THRESH = 0.50      # was 0.43
TOP2_MARGIN_MIN    = 0.08      # was 0.12
TOP2_RATIO_MAX     = 0.88      # was 0.80


if FAST_MODE:
    FRAME_DOWNSCALE = 1.0
    DETECT_EVERY_N_BASE = 10
    TRACKER_TYPE = "KCF"
    MODEL_NAME   = "Facenet"
    MAX_TEMPLATES_PER_ID = 6
    STABLE_FRAMES_AUTH   = 5
    COLOR_HOLD_FRAMES    = 8
    MIN_BBOX_AREA   = 70 * 70
    MIN_LAPLACE_VAR = 40.0
else:
    FRAME_DOWNSCALE = 1.0
    DETECT_EVERY_N_BASE = 6
    TRACKER_TYPE = "CSRT"
    MODEL_NAME   = "Facenet512"
    MAX_TEMPLATES_PER_ID = 6
    STABLE_FRAMES_AUTH   = 5
    COLOR_HOLD_FRAMES    = 12
    MIN_BBOX_AREA   = 80 * 80
    MIN_LAPLACE_VAR = 45.0

HEAVY_MIN_PERIOD_SEC   = 0.25
NO_FACE_BACKOFF_MAX_N  = 24
NO_FACE_BACKOFF_STEP   = 2

# staged capture
STAGE_LIST = [
    ("front",       4, "استعد لوضعية الوجه الأمامي"),
    ("right_side",  3, "استعد لوضعية الجانب الأيمن"),
    ("left_side",   3, "استعد لوضعية الجانب الأيسر"),
]
PRE_CAPTURE_COOLDOWN_SEC = 10.0
STAGE_COOLDOWN_SEC       = 10.0
STAGE_TIMEOUT_PER_STAGE  = 25.0
CAPTURE_IMAGE_INTERVAL   = 0.25

NEW_USER_PREFIX          = "user_"
MIRROR_WEBCAM            = False
CAPTURE_TRIGGER_COOLDOWN_SEC = 8.0
DIRECT_ENROLL_TO_AUTH    = True

CAPTURE_ONLY_WHEN_UNAUTHORIZED   = True
CAPTURE_MIN_DIST_FOR_NEW         = 0.55
CAPTURE_SUPPRESS_AFTER_AUTH_SEC  = 15.0

# TTS (Arabic)
ENABLE_TTS      = True
TTS_LANG        = "ar"
TTS_DEDUPE_SEC  = 4.0
TTS_CACHE_DIR   = Path("tts_cache")

# Name prompt
NAME_PROMPT_TIMEOUT_SEC = 12.0

# UI
DRAW_THICKNESS     = 2
FONT               = cv2.FONT_HERSHEY_SIMPLEX
DIST_SMOOTH_ALPHA  = 0.30
BBOX_SMOOTH_ALPHA  = 0.40
OVERLAY_TEXT       = (255,255,255)

# How much to zoom the window (display only)
UI_SCALE           = 1.5   # 1.0 = original; 2.0 = 2x bigger


# ================ GLOBAL STATE ================
DB_TEMPLATES: Dict[str, List[np.ndarray]] = {}

_current_identity: Optional[str] = None
_auth_streak = 0
_last_auth_time: Dict[str, float] = {}
_last_auth_seen_ts = 0.0
_last_repeat_tts_time = {}   # name -> timestamp of last “already logged” tts

_tracker = None
_tracker_bbox = None
_tracker_warned = False

_per_id_streak = {}  # identity -> streak of consecutive authorized frames

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

_last_secondary_overlays = []  # list of (x1, y1, x2, y2, label, (b,g,r))

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

# ----- External / API control flags -----
ENROLL_REQUESTED_FROM_API = False
ENROLL_LAST_API_REASON = ""
_face_thread = None  # used when running via FastAPI


# UI timed
_ui_lock = threading.Lock()
_countdown_active = False
_countdown_end_ts = 0.0
_countdown_label = ""
_flash_capture_until = 0.0
_name_prompt_banner_until = 0.0
_banner_until = 0.0
_banner_text = ""

def show_banner(msg: str, seconds: float = 2.5):
    global _banner_until, _banner_text
    with _ui_lock:
        _banner_text = msg
        _banner_until = time.time() + max(1.0, seconds)

def _ui_start_countdown(label: str, seconds: float):
    global _countdown_active, _countdown_end_ts, _countdown_label
    with _ui_lock:
        _countdown_label = label
        _countdown_end_ts = time.time() + max(1.0, seconds)
        _countdown_active = True

def _ui_stop_countdown():
    global _countdown_active
    with _ui_lock:
        _countdown_active = False

def _ui_flash_capture(duration: float = 0.6):
    global _flash_capture_until
    with _ui_lock:
        _flash_capture_until = time.time() + max(0.1, duration)

def _ui_show_name_prompt_banner(duration: float = NAME_PROMPT_TIMEOUT_SEC):
    global _name_prompt_banner_until
    with _ui_lock:
        _name_prompt_banner_until = time.time() + max(2.0, duration)

# ================ gTTS speaker ================
# ================ Simple Arabic TTS (gTTS + playsound) ================
# ================ Simple Arabic TTS (gTTS + playsound) ================
try:
    from gtts import gTTS
    from playsound import playsound
    _tts_available = True
except ImportError:
    print("[WARN] gTTS / playsound not installed → TTS disabled.")
    _tts_available = False
    ENABLE_TTS = False  # override global flag if missing deps

TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_last_tts = {"text": "", "ts": 0.0}
tts_lock = threading.Lock()


def _tts_get_path(text: str) -> str:
    """Return cached mp3 path for given text, generating it if needed."""
    key = hashlib.md5((TTS_LANG + "||" + text).encode("utf-8")).hexdigest()[:16]
    mp3_path = TTS_CACHE_DIR / f"{key}.mp3"

    if not mp3_path.exists():
        tts = gTTS(text=text, lang=TTS_LANG)
        tts.save(str(mp3_path))

    # IMPORTANT: fix path for Windows MCI (no backslashes)
    safe_path = str(mp3_path.resolve()).replace("\\", "/")
    return safe_path

def speak(text: str):
    if not ENABLE_TTS or not _tts_available:
        return
    """Non-blocking, cached Arabic TTS using gTTS + playsound (for general use)."""
    global _last_tts

    if not ENABLE_TTS:
        return

    now = time.time()
    last_text = _last_tts["text"]
    last_ts   = _last_tts["ts"]

    # de-duplicate same phrase within TTS_DEDUPE_SEC
    if text == last_text and (now - last_ts) < TTS_DEDUPE_SEC:
        return

    _last_tts["text"] = text
    _last_tts["ts"]   = now

    print("[SAY]", text)

    def _worker():
        try:
            safe_path = _tts_get_path(text)
            # 🔒 Only one audio at a time
            with tts_lock:
                playsound(safe_path)
        except Exception as e:
            print("[TTS ERROR]", e)

    threading.Thread(target=_worker, daemon=True).start()


def speak_sync(text: str):
    """Blocking TTS (used when we want the phrase to finish BEFORE continuing)."""
    if not ENABLE_TTS:
        return
    try:
        print("[SAY_SYNC]", text)
        safe_path = _tts_get_path(text)
        # 🔒 Also serialized with async speaks
        with tts_lock:
            playsound(safe_path)
    except Exception as e:
        print("[TTS ERROR SYNC]", e)

# ================ DB =================
def _db_connect():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(AUTH_DB_PATH))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_auth_db():
    with _db_connect() as con:
        # identities & templates
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
        # attendance (per-day Arrival/Departure)
        con.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            name TEXT NOT NULL,
            status TEXT NOT NULL,                 -- 'Arrival' or 'Departure'
            ts_iso TEXT NOT NULL,
            ts_unix REAL NOT NULL,
            distance REAL,
            created_at REAL NOT NULL,
            FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_attendance_ts ON attendance(ts_unix)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_attendance_name ON attendance(name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_templates_pid ON templates(person_id)")

        # NEW: new_users_log table (log enrollments)
        con.execute("""
        CREATE TABLE IF NOT EXISTS new_users_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            name TEXT NOT NULL,
            ts_iso TEXT NOT NULL,
            ts_unix REAL NOT NULL,
            FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
        )""")

def db_get_person_id_by_name(name: str) -> Optional[int]:
    with _db_connect() as con:
        r = con.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
        return int(r[0]) if r else None

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
    if not STORE_IMAGES_IN_DB:
        return
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        return
    with _db_connect() as con:
        con.execute(
            "INSERT INTO images(person_id, stage, img, created_at) VALUES(?, ?, ?, ?)",
            (person_id, stage, sqlite3.Binary(buf.tobytes()), time.time())
        )

def db_templates_dict() -> Dict[str, List[np.ndarray]]:
    with _db_connect() as con:
        rows = con.execute("""
            SELECT p.name, t.emb FROM templates t
            JOIN persons p ON p.id = t.person_id
        """).fetchall()
    out: Dict[str, List[np.ndarray]] = {}
    for name, emb_blob in rows:
        e = np.frombuffer(emb_blob, dtype=np.float32)
        e = _l2_normalize(e)
        out.setdefault(name, []).append(e)
    for name, lst in out.items():
        if len(lst) > MAX_TEMPLATES_PER_ID:
            step = max(1, len(lst)//MAX_TEMPLATES_PER_ID)
            out[name] = lst[::step][:MAX_TEMPLATES_PER_ID]
    return out

def migrate_folder_to_sqlite_if_needed(model_pair):
    with _db_connect() as con:
        n = con.execute("SELECT COUNT(1) FROM persons").fetchone()[0]
    if n > 0:
        return
    if not AUTHORIZED_DIR.exists():
        return
    print("[MIGRATE] importing authorized_faces/ → sqlite…")
    folder_db = load_authorized_db(model_pair, AUTHORIZED_DIR)
    for name, embs in folder_db.items():
        pid = db_add_person(name)
        for e in embs:
            db_add_template(pid, e)
    print("[MIGRATE] done.")

# Attendance logging (DB)
def log_event(name: str, status: str, distance: Optional[float]) -> bool:
    """
    Log exactly ONE Arrival and ONE Departure per person per calendar day.
    If an entry for (name, status, today) already exists, we still SAY a welcome/goodbye
    message, but we don't insert a new DB row.
    """
    ts = datetime.now()
    ts_iso = ts.isoformat(timespec="seconds")
    ts_unix = time.time()
    today_str = ts.strftime("%Y-%m-%d")  # for daily uniqueness
    pid = db_get_person_id_by_name(name)

    try:
        with _db_connect() as con:
            # Check if this person already has this status today
            already = con.execute("""
                SELECT 1 FROM attendance
                WHERE name = ?
                  AND status = ?
                  AND substr(ts_iso, 1, 10) = ?
                LIMIT 1
            """, (name, status, today_str)).fetchone()

            if already:
                # Banner + logging
                show_banner(f"{name} already has {status} logged today")
                print(f"[INFO] Skip log_event: {name} already has {status} for {today_str}")

                # 🔊 TTS: special message when attendance already exists
                # --- Only play this message if at least 5 seconds passed since last one for this user ---
                global _last_repeat_tts_time
                now_ts = time.time()
                last_ts = _last_repeat_tts_time.get((name, status), 0)

                if (now_ts - last_ts) >= 20.0:
                    if status == "Arrival":
                        speak(f"مرحباً يا {name}، تم تسجيل حضورك بالفعل اليوم.")
                    else:
                        speak(f"إلى اللقاء يا {name}، تم تسجيل انصرافك بالفعل اليوم.")
                    _last_repeat_tts_time[(name, status)] = now_ts


                return False  # no new DB insert

            # Otherwise, insert a new record
            con.execute("""
                INSERT INTO attendance(person_id, name, status, ts_iso, ts_unix, distance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pid,
                name,
                status,
                ts_iso,
                ts_unix,
                None if distance is None else float(distance),
                time.time()
            ))

        show_banner(f"Logged: {name} → {status}")
        print(f"[INFO] Attendance logged (DB) for {name}: {status}")

        # 🔊 TTS feedback (Arabic) + welcome/goodbye for first time today
        if status == "Arrival":
            speak(f"تم تسجيل الحضور لِـ {name}")
            speak(f"مرحباً يا {name}")
        else:
            speak(f"تم تسجيل الانصراف لِـ {name}")
            speak(f"إلى اللقاء يا {name}")

        return True

    except Exception as e:
        print(f"[WARN] Attendance DB write failed: {e}")
        return False

# New users logging
def log_new_user(name: str, person_id: Optional[int]):
    ts = datetime.now()
    ts_iso = ts.isoformat(timespec="seconds")
    ts_unix = time.time()
    try:
        with _db_connect() as con:
            con.execute("""
                INSERT INTO new_users_log(person_id, name, ts_iso, ts_unix)
                VALUES (?, ?, ?, ?)
            """, (person_id, name, ts_iso, ts_unix))
        print(f"[INFO] New user logged: {name} (person_id={person_id})")
        show_banner(f"New user added: {name}")
        speak(f"تم إضافة مستخدم جديد باسم {name}")
    except Exception as e:
        print(f"[WARN] New user log failed: {e}")

# ================ DeepFace model ================
def build_model():
    if DeepFace is None:
        raise RuntimeError("DeepFace not installed. Please install deepface==0.0.79 (or similar).")
    return (DeepFace.build_model(MODEL_NAME), None)

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return float(1.0 - np.dot(a, b))


# ============ PREPROCESSING HELPERS (lighting + rotation) ============

def _normalize_gray_lighting(gray: np.ndarray) -> np.ndarray:
    """
    Improve contrast of grayscale image using CLAHE.
    Helps detection under poor / uneven lighting.
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        return gray_eq
    except Exception:
        return gray


def _auto_gamma_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Simple auto-gamma to brighten dark faces and slightly dim over-bright ones.
    """
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return bgr

    mean_val = float(gray.mean())

    # Decide gamma based on brightness
    if mean_val < 60:
        gamma = 0.6      # strongly brighten
    elif mean_val < 90:
        gamma = 0.8      # mildly brighten
    elif mean_val > 200:
        gamma = 1.4      # strongly darken
    elif mean_val > 170:
        gamma = 1.2      # mildly darken
    else:
        gamma = 1.0      # leave as is

    if abs(gamma - 1.0) < 1e-3:
        return bgr

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(bgr, table)


def preprocess_face_crop(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess the face crop to be more robust to lighting:
    - CLAHE on L channel in LAB space
    - auto-gamma for global brightness
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    try:
        # CLAHE on L channel
        lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Auto-gamma
        bgr_gamma = _auto_gamma_bgr(bgr_eq)
        return bgr_gamma
    except Exception:
        return bgr_crop


def _rotate_crop(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a face crop by a small angle around its center.
    Used to handle tilted heads / slightly wrong camera angle.
    """
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return rotated



def embed_bgr_crop(model_pair, bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Embed a single face crop with lighting normalization.
    This is used by enrollment and by the multi-orientation helper.
    """
    if bgr is None or bgr.size == 0:
        return None
    try:
        # 🔧 lighting normalization first
        bgr_proc = preprocess_face_crop(bgr)

        rgb = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)
        reps = DeepFace.represent(
            img_path=rgb,
            model_name=MODEL_NAME,
            detector_backend="skip",   # we already have the crop
            enforce_detection=False
        )
        if not reps:
            return None
        e = np.array(reps[0]["embedding"], dtype=np.float32)
        return _l2_normalize(e)
    except Exception as e:
        print(f"[WARN] embedding failed: {e}")
        return None

# ================ Detection (cascade) ================
FRONTAL_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def _dynamic_min_size(w: int, h: int) -> Tuple[int, int]:
    m = max(32, int(min(w, h) * 0.10))
    m = min(m, max(32, min(w, h)))
    return (m, m)

def _detect_with_cascade(gray, cascade, min_size):
    try:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=8,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=min_size
        )
        return faces
    except Exception as e:
        print("[DETECT] error:", e)
        return ()

def _bbox_from_faces(faces):
    if len(faces) == 0:
        return None
    x,y,ww,hh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    return (x,y,x+ww,y+hh)

def detect_largest_face_bbox(bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    if bgr is None or bgr.size == 0:
        return None
    h,w = bgr.shape[:2]
    if w < 32 or h < 32:
        return None
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None

    # 🔧 make detection more robust under poor lighting
    gray = _normalize_gray_lighting(gray)

    mn = _dynamic_min_size(w,h)


    faces = _detect_with_cascade(gray, FRONTAL_CASCADE, mn)
    if len(faces):
        return _bbox_from_faces(faces)

    faces = _detect_with_cascade(gray, PROFILE_CASCADE, mn)
    if len(faces):
        return _bbox_from_faces(faces)

    gray_flipped = cv2.flip(gray, 1)
    faces = _detect_with_cascade(gray_flipped, PROFILE_CASCADE, mn)
    if len(faces):
        x,y,ww,hh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        x1 = w - (x + ww); y1 = y; x2 = w - x; y2 = y + hh
        return (x1,y1,x2,y2)

    return None

def detect_multi_face_bboxes(bgr: np.ndarray, max_faces: int = 5) -> List[Tuple[int,int,int,int]]:
    """
    Detect up to max_faces faces (front + left + right profiles) and return boxes (x1, y1, x2, y2).
    """
    if bgr is None or bgr.size == 0:
        return []
    h, w = bgr.shape[:2]
    if w < 32 or h < 32:
        return []

    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return []

    # 🔧 lighting normalization for multi-face detection
    gray = _normalize_gray_lighting(gray)

    mn = _dynamic_min_size(w, h)


    boxes: List[Tuple[int,int,int,int]] = []

    # 1) frontal faces
    faces_f = _detect_with_cascade(gray, FRONTAL_CASCADE, mn)
    for (x, y, ww, hh) in faces_f:
        boxes.append((x, y, x + ww, y + hh))

    # 2) right profiles
    faces_r = _detect_with_cascade(gray, PROFILE_CASCADE, mn)
    for (x, y, ww, hh) in faces_r:
        boxes.append((x, y, x + ww, y + hh))

    # 3) left profiles (flip)
    gray_flipped = cv2.flip(gray, 1)
    faces_l = _detect_with_cascade(gray_flipped, PROFILE_CASCADE, mn)
    for (x, y, ww, hh) in faces_l:
        x1 = w - (x + ww)
        y1 = y
        x2 = w - x
        y2 = y + hh
        boxes.append((x1, y1, x2, y2))

    if not boxes:
        # fallback: largest single face using the old routine
        one = detect_largest_face_bbox(bgr)
        return [one] if one is not None else []

    # remove near-duplicate boxes (simple overlap filter)
    merged = []
    for b in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        x1, y1, x2, y2 = b
        area_b = (x2 - x1) * (y2 - y1)
        keep = True
        for (mx1, my1, mx2, my2) in merged:
            inter_x1 = max(x1, mx1)
            inter_y1 = max(y1, my1)
            inter_x2 = min(x2, mx2)
            inter_y2 = min(y2, my2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            if inter_area / float(area_b + 1e-6) > 0.6:
                keep = False
                break
        if keep:
            merged.append(b)

    merged = sorted(merged, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    if max_faces > 0 and len(merged) > max_faces:
        merged = merged[:max_faces]

    return merged

def detect_face_for_stage(bgr: np.ndarray, stage_name: str) -> Optional[Tuple[int,int,int,int]]:
    # stage-aware detection (front/profile/left via flipped profile)
    if bgr is None or bgr.size == 0:
        return None
    h,w = bgr.shape[:2]
    if w < 32 or h < 32:
        return None
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None

    # 🔧 more robust stage detection in bad lighting
    gray = _normalize_gray_lighting(gray)

    mn = _dynamic_min_size(w,h)


    if stage_name == "front":
        faces = _detect_with_cascade(gray, FRONTAL_CASCADE, mn)
        if len(faces):
            return _bbox_from_faces(faces)
        return detect_largest_face_bbox(bgr)

    if stage_name == "right_side":
        faces = _detect_with_cascade(gray, PROFILE_CASCADE, mn)
        if len(faces):
            return _bbox_from_faces(faces)
        return detect_largest_face_bbox(bgr)

    if stage_name == "left_side":
        gray_flipped = cv2.flip(gray, 1)
        faces = _detect_with_cascade(gray_flipped, PROFILE_CASCADE, mn)
        if len(faces):
            x,y,ww,hh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            x1 = w - (x + ww); y1 = y; x2 = w - x; y2 = y + hh
            return (x1,y1,x2,y2)
        return detect_largest_face_bbox(bgr)

    return detect_largest_face_bbox(bgr)

def is_crop_usable(bgr_crop: np.ndarray) -> bool:
    if bgr_crop is None or bgr_crop.size == 0:
        return False

    h, w = bgr_crop.shape[:2]
    if (w * h) < MIN_BBOX_AREA:
        return False

    g = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    # ---- NEW: reject over-bright / over-dark blobs (like lamps) ----
    mean_val = float(g.mean())
    # tune these numbers if needed
    if mean_val > 220:      # too bright → likely lamp / light source
        return False
    if mean_val < 35:       # too dark → noisy shadows
        return False

    # optional: reject crops with too many saturated pixels
    sat_ratio = np.count_nonzero(g > 245) / (w * h)
    if sat_ratio > 0.30:    # >30% of pixels are pure white
        return False

    # existing sharpness / blur check
    if cv2.Laplacian(g, cv2.CV_64F).var() < MIN_LAPLACE_VAR:
        return False

    return True


# ================ Tracker / smoothing ================
def _create_tracker(kind="KCF"):
    global _tracker_warned
    kind = kind.upper()

    # 1) Try legacy module if it really has the tracker
    if hasattr(cv2, "legacy"):
        if kind == "KCF" and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        if kind == "CSRT" and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()

    # 2) Try non-legacy API if available
    if kind == "KCF" and hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if kind == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    # 3) Fallbacks: MIL / MOSSE if present
    if hasattr(cv2, "TrackerMIL_create"):
        if not _tracker_warned:
            _tracker_warned = True
        return cv2.TrackerMIL_create()

    if hasattr(cv2, "TrackerMOSSE_create"):
        if not _tracker_warned:
            print("[WARN] KCF/CSRT not available, falling back to MOSSE tracker.")
            _tracker_warned = True
        return cv2.TrackerMOSSE_create()

    # 4) Nothing available → disable tracker
    if not _tracker_warned:
        print("[WARN] No suitable OpenCV tracker found; tracker will be disabled.")
        _tracker_warned = True
    return None

def ema(prev, new, alpha):
    return new if prev is None else (alpha*new + (1-alpha)*prev)

def ema_bbox(prev, new, alpha):
    nx1, ny1, nx2, ny2 = map(float, new)
    if prev is None:
        return (nx1, ny1, nx2, ny2)
    px1, py1, px2, py2 = prev
    return (alpha*nx1 + (1-alpha)*px1,
            alpha*ny1 + (1-alpha)*py1,
            alpha*nx2 + (1-alpha)*px2,
            alpha*ny2 + (1-alpha)*py2)

# ================ Recognition ================
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

    # 1) If only ONE identity in DB, be extra strict
    SINGLE_ID_MAX_ACCEPT = 0.32
    if len(pairs) == 1 and best_dist > SINGLE_ID_MAX_ACCEPT:
        return None

    # 2) Global "too far from everyone" cutoff
    HARD_UNKNOWN_THRESH = 0.55
    if best_dist > HARD_UNKNOWN_THRESH:
        return None

    return best_name, best_dist, second

def best_match_for_crop(model_pair, db, bgr_crop: np.ndarray):
    """
    Try a few small rotations of the face crop and choose the orientation
    that gives the best (smallest) distance to the DB.

    Returns (best_name, best_dist, second_best_dist) or None.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    angles = [0.0, -12.0, 12.0]   # you can add e.g. -20, +20 if needed
    candidates = []

    for ang in angles:
        rotated = bgr_crop if ang == 0.0 else _rotate_crop(bgr_crop, ang)
        if rotated is None or rotated.size == 0:
            continue

        emb = embed_bgr_crop(model_pair, rotated)
        if emb is None:
            continue

        m = _match(db, emb)
        if m is None:
            continue

        best_name, best_dist, second = m
        candidates.append((best_name, best_dist, second))

    if not candidates:
        return None

    # Pick orientation with minimum best_dist
    return min(candidates, key=lambda t: t[1])


def recognize_heavy(model_pair, db, bgr, max_faces: int = 5):
    """
    Multi-face recognition.
    Returns a list of (bbox, identity, dist, is_auth_fast, secs)
    for up to max_faces faces.
    """
    t0 = time.perf_counter()
    boxes = detect_multi_face_bboxes(bgr, max_faces=max_faces)
    if not boxes:
        return []   # instead of None

    results = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        crop = bgr[y1:y2, x1:x2]
        if not is_crop_usable(crop):
            continue

        match = best_match_for_crop(model_pair, db, crop)
        if match is None:
            continue

        best_name, best_dist, second = match

        clear_margin = (second - best_dist) >= TOP2_MARGIN_MIN
        clear_ratio  = (best_dist / max(second, 1e-9)) <= TOP2_RATIO_MAX
        accept       = best_dist <= ACCEPT_DIST_THRESH
        is_auth_fast = accept and clear_margin and clear_ratio

        # 🔍 DEBUG: see how good/bad the match is
        print(
            f"[REC] name={best_name} d={best_dist:.3f} second={second:.3f} "
            f"accept={accept} margin={second - best_dist:.3f} ratio={best_dist / max(second, 1e-9):.3f}"
        )

        secs = time.perf_counter() - t0
        results.append((bbox, best_name, best_dist, is_auth_fast, secs))

    return results

# ================ Legacy folder -> embeddings cache ================
def _safe_stat(path: str):
    try:
        st = os.stat(path); return (st.st_mtime, st.st_size)
    except Exception:
        return (0.0, -1)

def _scan_manifest(directory: Path):
    images_by_name = {}
    for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
        for p in directory.glob(ext):
            images_by_name.setdefault(p.stem, []).append(str(p))
    for sub in [d for d in directory.glob("*") if d.is_dir()]:
        paths=[]
        for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
            paths.extend([str(x) for x in sub.glob(ext)])
        if paths:
            images_by_name[sub.name] = paths
    manifest = {}
    for name, paths in images_by_name.items():
        manifest[name] = [(p,)+_safe_stat(p) for p in sorted(paths)]
    return manifest

def _current_cache_settings():
    return {"MODEL_NAME": MODEL_NAME, "MAX_TEMPLATES_PER_ID": MAX_TEMPLATES_PER_ID}

def _load_cache():
    if not CACHE_ENABLED or not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def _manifests_equal(a,b):
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        if len(a[k]) != len(b.get(k, [])):
            return False
        for (pa,ma,sa), (pb,mb,sb) in zip(a[k], b[k]):
            if pa != pb or ma != mb or sa != sb:
                return False
    return True

def _save_cache(db, manifest):
    if not CACHE_ENABLED:
        return
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: [e.astype(np.float32) for e in v] for k,v in db.items()}
        blob = {"settings": _current_cache_settings(),
                "manifest": manifest, "db": serializable, "saved_at": time.time()}
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CACHE] saved in {CACHE_PATH}")
    except Exception as e:
        print("[CACHE] save failed:", e)

def load_authorized_db(model_pair, directory: Path) -> Dict[str, List[np.ndarray]] :
    manifest_now = _scan_manifest(directory)
    cache = _load_cache()
    if cache is not None:
        try:
            if cache.get("settings") == _current_cache_settings() and _manifests_equal(cache.get("manifest", {}), manifest_now):
                db_cached = cache.get("db", {})
                if isinstance(db_cached, dict) and db_cached:
                    db = {k: [np.array(e, dtype=np.float32) for e in v] for k,v in db_cached.items()}
                    print(f"[CACHE] Loaded {len(db)} identities from cache.")
                    return db
        except Exception:
            pass

    images_by_name = {}
    for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
        for p in directory.glob(ext):
            images_by_name.setdefault(p.stem, []).append(str(p))
    for sub in [d for d in directory.glob("*") if d.is_dir()]:
        paths=[]
        for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
            paths.extend([str(x) for x in sub.glob(ext)])
        if paths:
            images_by_name[sub.name] = paths

    if not images_by_name:
        print(f"[WARN] No images in {directory}, everyone will be NOT AUTHORIZED.")
        return {}

    db = {}
    for name, paths in images_by_name.items():
        imgs = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                imgs.append(img)
        if not imgs:
            continue
        embs=[]
        for img in imgs:
            bbox = detect_largest_face_bbox(img)
            if bbox is None:
                continue
            x1,y1,x2,y2 = bbox
            crop = img[y1:y2, x1:x2]
            if not is_crop_usable(crop):
                continue
            e = embed_bgr_crop(model_pair, crop)
            if e is not None:
                embs.append(e)
        if embs:
            if len(embs) > MAX_TEMPLATES_PER_ID:
                step = max(1, len(embs)//MAX_TEMPLATES_PER_ID)
                embs = embs[::step][:MAX_TEMPLATES_PER_ID]
            db[name] = embs
            print(f"[OK] authorized: {name} templates={len(embs)}")
    _save_cache(db, manifest_now)
    return db

# ================ Capture Worker ================
class CaptureWorker:
    def __init__(self):
        self._thread = None

    def start_capture(self):
        if self._thread is not None and self._thread.is_alive():
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self):
        global _capture_in_progress
        try:
            with _capture_lock:
                if _capture_in_progress:
                    return
                _capture_in_progress = True

            # Session metadata (no folders yet, just an ID)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = NEW_USER_PREFIX + ts
            print(f"[CAPTURE] starting new session id={default_name}")

            # Preparation delay (no writing to disk)
            speak("جارٍ بدء التقاط صور المستخدم الجديد، من فضلك انتظر.")
            t_end = time.time() + PRE_CAPTURE_COOLDOWN_SEC
            while time.time() < t_end:
                time.sleep(0.05)

            captured_images = []  # list of (stage_name, crop np.ndarray)
            total_saved = 0

            for idx, (stage_name, stage_count, stage_prompt) in enumerate(STAGE_LIST, start=1):
                # 🔊 BLOCKING TTS: finish "be ready..." first, THEN show countdown
                speak_sync(stage_prompt)

                _ui_start_countdown(f"{stage_name.replace('_',' ').title()} in", 3)
                time.sleep(3.2)
                _ui_stop_countdown()

                saved = 0
                last_save = 0.0
                deadline = time.time() + STAGE_TIMEOUT_PER_STAGE
                while saved < stage_count and time.time() < deadline:
                    with _latest_frame_lock:
                        f = None if _latest_frame_for_capture is None else _latest_frame_for_capture.copy()
                    if f is None or f.size == 0:
                        time.sleep(0.05)
                        continue

                    bbox = detect_face_for_stage(f, stage_name)
                    if bbox is None:
                        time.sleep(0.04)
                        continue

                    x1, y1, x2, y2 = bbox
                    h, w = f.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        time.sleep(0.04)
                        continue

                    now = time.time()
                    if (now - last_save) < CAPTURE_IMAGE_INTERVAL:
                        time.sleep(0.03)
                        continue

                    pad = 0.12
                    dx = int((x2 - x1) * pad)
                    dy = int((y2 - y1) * pad)
                    x1p = max(0, x1 - dx)
                    y1p = max(0, y1 - dy)
                    x2p = min(w - 1, x2 + dx)
                    y2p = min(h - 1, y2 + dy)
                    crop = f[y1p:y2p, x1p:x2p]
                    if crop is None or crop.size == 0:
                        time.sleep(0.03)
                        continue

                    # Visual flash only, no disk write
                    _ui_flash_capture(0.6)
                    captured_images.append((stage_name, crop.copy()))
                    saved += 1
                    total_saved += 1
                    last_save = now
                    print(f"[CAPTURE] [{stage_name}] captured image #{saved}")

                if idx < len(STAGE_LIST):
                    t_end = time.time() + STAGE_COOLDOWN_SEC
                    while time.time() < t_end:
                        time.sleep(0.05)

            if total_saved == 0:
                print("[CAPTURE] nothing captured.")
                speak("لم يتم حفظ أي صور. سيتم إلغاء التسجيل.")
                return

            # Push an in-memory session into the queue
            session = {
                "timestamp": ts,
                "default_name": default_name,
                "images": captured_images,  # list of (stage_name, crop)
            }
            with _pending_staging_lock:
                _pending_staging_queue.append(session)

            print(f"[CAPTURE] ready: session={default_name} (saved={total_saved})")
            speak("تم الانتهاء من التقاط الصور. من فضلك أدخل اسمك في نافذة الأوامر.")

        except Exception as e:
            print("[CAPTURE] error:", e)
            traceback.print_exc()
        finally:
            with _capture_lock:
                _capture_in_progress = False

# ================ Enrollment finalize ================
def embed_from_session(model_pair, images: list) -> Optional[np.ndarray]:
    """
    Given a list of (stage_name, img_bgr) from memory,
    return the first usable embedding (for duplicate check).
    """
    for stage_name, img in images:
        if img is None or img.size == 0:
            continue
        bbox = detect_largest_face_bbox(img)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        if not is_crop_usable(crop):
            continue
        e = embed_bgr_crop(model_pair, crop)
        if e is not None:
            return e
    return None

def duplicate_check(emb) -> Optional[Tuple[str,float]]:
    best=None;best_d=1e9
    for name, refs in DB_TEMPLATES.items():
        if not refs: continue
        d=min(cosine_distance(emb,r) for r in refs)
        if d<best_d: best_d=d; best=name
    return (best,best_d) if best is not None else None

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
    _ui_show_name_prompt_banner(NAME_PROMPT_TIMEOUT_SEC)
    speak("اكتب اسمك في نافذة الأوامر ثم اضغط إنتر.")
    entered = _timed_input(
        f"New user name (default '{default_name}', {NAME_PROMPT_TIMEOUT_SEC:.0f}s): ",
        NAME_PROMPT_TIMEOUT_SEC,
        default_name
    )
    entered = re.sub(r"\s+","_",entered)
    entered = re.sub(r"[^A-Za-z0-9_\-]","",entered)[:64]
    return entered or default_name

def finalize_one(model_pair, session: dict):
    global _request_db_reload
    if not session or "images" not in session:
        return

    images = session["images"]
    if not images:
        return

    print(f"[ENROLL] finalizing in-memory session: {session.get('default_name', '?')}")

    # 1) Get one embedding for duplicate check
    emb = embed_from_session(model_pair, images)
    if emb is None:
        print("[ENROLL] no usable embedding from captured images.")
        speak("لم يتم العثور على وجه مناسب في الصور. سيتم إلغاء التسجيل.")
        return

    # 2) Duplicate check
    dup = duplicate_check(emb)
    if dup is not None:
        name, d = dup
        print(f"[ENROLL] duplicate '{name}' d={d:.3f}")
        if d <= ACCEPT_DIST_THRESH:
            speak("هذا الوجه مسجل بالفعل، لن يتم إنشاء مستخدم جديد.")
            return


    # 3) Ask for name
    default_name = session.get(
        "default_name",
        NEW_USER_PREFIX + session.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    final_name = prompt_name(default_name)

    # 3.5) Block re-enrollment by name
    existing_pid = db_get_person_id_by_name(final_name)
    if existing_pid is not None:
        print(f"[ENROLL] name '{final_name}' already exists in DB, aborting extra enrollment.")
        speak("هذا الاسم مسجل بالفعل، لن يتم إنشاء مستخدم جديد.")
        return

    try:
        # 4) Create person in DB
        pid = db_add_person(final_name)
        embs_added = 0


        # 5) Create folder ONLY NOW (after successful name entry)
        if DIRECT_ENROLL_TO_AUTH:
            person_dir = AUTHORIZED_DIR / final_name
        else:
            person_dir = Path("staging_enroll") / final_name
        person_dir.mkdir(parents=True, exist_ok=True)

        img_idx = 0
        for stage_name, img in images:
            if img is None or img.size == 0:
                continue

            bbox = detect_largest_face_bbox(img)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            if not is_crop_usable(crop):
                continue

            e = embed_bgr_crop(model_pair, crop)
            if e is None:
                continue

            # Add template to SQLite
            db_add_template(pid, e)
            embs_added += 1

            # Save physical image file AFTER everything is valid
            img_idx += 1
            out_path = person_dir / f"{final_name}_{stage_name}_{img_idx:02d}.jpg"
            try:
                cv2.imwrite(str(out_path), img)
            except Exception as ee:
                print(f"[ENROLL] warning: failed to save {out_path}: {ee}")

            # Optional: also store in DB if enabled
            if STORE_IMAGES_IN_DB:
                db_add_image(pid, stage_name, crop)

            if embs_added >= MAX_TEMPLATES_PER_ID:
                break

        print(f"[ENROLL] finalized in SQLite for '{final_name}' (templates: {embs_added})")

        # Log new user in separate table + TTS
        log_new_user(final_name, pid)

    except Exception as e:
        print("[ENROLL] SQLite finalize failed:", e)

    # 6) Trigger DB reload for the recognizer
    with _request_db_reload_lock:
        _request_db_reload = True

def finalize_all(model_pair):
    while True:
        with _pending_staging_lock:
            if not _pending_staging_queue:
                return
            session = _pending_staging_queue.popleft()
        finalize_one(model_pair, session)
# ------------------- ENROLLMENT GATING -------------------
def enroll_allowed_now() -> Tuple[bool, str]:
    """
    Decide whether we allow starting a NEW capture session now.

    TEMPORARILY RELAXED:
    - We only block if capture is already in progress
      or if global cooldown (8 seconds) is not finished.
    - We DO NOT block based on _current_identity / _smoothed_dist /
      CAPTURE_ONLY_WHEN_UNAUTHORIZED right now.
    """
    if _capture_in_progress:
        return False, "capture already in progress"

    now_wall = time.time()
    if (now_wall - _last_capture_trigger_ts) < CAPTURE_TRIGGER_COOLDOWN_SEC:
        return False, "global capture cooldown"

    # relaxed: always allow after cooldown
    return True, "ok"


def _update_streak_and_maybe_log(identity: str,
                                 is_authorized: bool,
                                 dist: Optional[float]):
    """
    Per-identity streak logic.
    Every authorized frame increments that person's streak.
    When streak >= STABLE_FRAMES_AUTH and cooldown passes, we log Arrival/Departure.
    """
    global _per_id_streak, _last_auth_time, _last_auth_seen_ts

    if not identity or identity == "Unknown":
        return

    # Reset streak if not authorized this frame
    if not is_authorized:
        _per_id_streak[identity] = 0
        return

    # Authorized this frame → increment streak
    prev = _per_id_streak.get(identity, 0) + 1
    _per_id_streak[identity] = prev

    if prev < STABLE_FRAMES_AUTH:
        return  # not stable yet

    # Streak is stable → check cooldown
    now = time.time()
    last = _last_auth_time.get(identity, 0.0)
    if (now - last) < 45.0:
        # still in cooldown
        _per_id_streak[identity] = 0
        return

    # Decide Arrival vs Departure
    status = "Arrival" if datetime.now().hour < 18 else "Departure"

    # Try to log; log_event will also enforce 1 Arrival + 1 Departure per day
    if log_event(identity, status, dist):
        _last_auth_time[identity] = now
        _last_auth_seen_ts = now

    # reset streak after firing
    _per_id_streak[identity] = 0

def request_enroll_from_api():
    """
    Called from FastAPI instead of pressing 'g'.
    Returns (ok: bool, reason: str)
    """
    global ENROLL_REQUESTED_FROM_API, ENROLL_LAST_API_REASON

    # same gating used for keyboard
    allowed, reason = enroll_allowed_now()
    print(f"[API] enroll_allowed_now -> allowed={allowed}, reason='{reason}'")  # <<< DEBUG

    if not allowed:
        return False, reason

    if _capture_in_progress:
        return False, "capture already in progress"

    ENROLL_REQUESTED_FROM_API = True
    ENROLL_LAST_API_REASON = "requested_from_api"
    print("[API] Enrollment requested from HR panel.")
    return True, "ok"



def start_system_in_background():
    """
    For FastAPI: start main() in a background thread.
    Do NOT call main() from the FastAPI request handler directly.
    """
    global _face_thread

    if _face_thread is not None and _face_thread.is_alive():
        print("[API] Face system already running.")
        return

    def _runner():
        try:
            main()
        except Exception as e:
            print("[API] Face system crashed:", e)
            traceback.print_exc()

    _face_thread = threading.Thread(target=_runner, daemon=True)
    _face_thread.start()
    print("[API] Face system started in background thread.")


# ================ Main =================
def main():
    global _last_heavy_submit_ts, _detect_every_n, _no_face_cycles
    global _current_identity, _auth_streak, _smoothed_dist, _smoothed_bbox
    global _last_detect_recog_s, _last_auth_latency_s, _pending_auth_identity, _pending_auth_start_ts
    global _latest_frame_for_capture, _last_auth_seen_ts, _last_capture_trigger_ts, _last_capture_frame_idx
    global DB_TEMPLATES, _tracker, _tracker_bbox, _hold_counter, _last_draw_color, _last_draw_label
    global _request_db_reload
    global _countdown_active, _countdown_end_ts, _countdown_label, _flash_capture_until, _name_prompt_banner_until
    global _banner_until, _banner_text
    global _last_secondary_overlays, _per_id_streak
    global ENROLL_REQUESTED_FROM_API, ENROLL_LAST_API_REASON   # ← add this line

    model_pair = build_model()
    init_auth_db()
    migrate_folder_to_sqlite_if_needed(model_pair)
    DB_TEMPLATES = db_templates_dict()

    win_name = "Face System (SQLite, press 'g' to enroll)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)   # ~YouTube size



    # Quick sanity check: are cascades loaded?
    print("[DEBUG] Haar path:", cv2.data.haarcascades)
    print("[DEBUG] FRONTAL_CASCADE empty:", FRONTAL_CASCADE.empty())
    print("[DEBUG] PROFILE_CASCADE empty:", PROFILE_CASCADE.empty())

    # ---------- Camera open helper ----------
    def open_capture():
        if USE_IP_CAMERA:
            # FFMPEG backend is usually more stable for RTSP
            cap_ = cv2.VideoCapture(IP_CAM_URL, cv2.CAP_FFMPEG)
        else:
            cap_ = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

        # Reduce buffering/latency for RTSP streams
        cap_.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap_

    cap = open_capture()
    if not cap.isOpened():
        print("[ERROR] camera open failed at startup")
        return

# --------------------------------------------------------- #    
    ########
        #stream url
    #######
   ## cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) 
    ##if not cap.isOpened():
      ##  print("[ERROR] camera open failed")
        ##return

#    cap = cv2.VideoCapture(stream_url) 
#    if not cap.isOpened():
#        print("[ERROR] camera open failed")
#        return
# --------------------------------------------------------- #   
    
    # recognizer thread
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
                    self.res = recognize_heavy(self.model, DB_TEMPLATES, f)
                except Exception as e:
                    print("[WARN] recognizer failed:", e)
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

    print("[INFO] running. Press 'g' to enroll. 'q' to quit.")
    speak("النظام يعمل الآن. يمكنك الضغط على حرف جي من لوحة المفاتيح لتسجيل مستخدم جديد.")
    loop_ts = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()

            # RTSP streams drop sometimes; try to recover instead of killing the app
            if not ok or frame is None:
                print("[WARN] camera read failed, trying to reopen stream...")
                cap.release()
                time.sleep(1.0)
                cap = open_capture()
                if not cap.isOpened():
                    print("[ERROR] failed to reopen camera, exiting.")
                    break
                continue


            now = time.perf_counter()
            dt = now - loop_ts
            loop_ts = now
            fps = (1.0 / dt) if dt > 0 else 0.0

            frame_disp = cv2.resize(
                frame,
                None,
                fx=FRAME_DOWNSCALE,
                fy=FRAME_DOWNSCALE
            ) if FRAME_DOWNSCALE != 1.0 else frame

            if MIRROR_WEBCAM:
                frame_disp = cv2.flip(frame_disp, 1)

            with _latest_frame_lock:
                _latest_frame_for_capture = frame.copy()

            frame_idx += 1

            # 1) Recognition cadence
            run_heavy_due_to_cadence = (frame_idx % _detect_every_n) == 0 or (_tracker is None)
            run_heavy_due_to_period  = (time.time() - _last_heavy_submit_ts) >= HEAVY_MIN_PERIOD_SEC

            if run_heavy_due_to_cadence and run_heavy_due_to_period:
                _last_heavy_submit_ts = time.time()
                recog.submit(frame_disp)
                rec = recog.get()

                if rec is None:
                    # No fresh result yet → keep previous tracker/labels as-is
                    pass

                elif not rec:
                    # Worker ran, but found 0 faces
                    _hold_counter = max(0, _hold_counter - 1)
                    _no_face_cycles = min(_no_face_cycles + 1, 50)
                    _detect_every_n = min(
                        DETECT_EVERY_N_BASE + _no_face_cycles * NO_FACE_BACKOFF_STEP,
                        NO_FACE_BACKOFF_MAX_N
                    )
                    _last_secondary_overlays = []
                    _current_identity = None
                    _auth_streak = 0

                else:
                    _no_face_cycles = 0
                    _detect_every_n = DETECT_EVERY_N_BASE

                    # --------- CHOOSE STABLE PRIMARY FACE ----------
                    primary_idx = 0
                    if _current_identity is not None:
                        for i, (bbox_i, id_i, dist_i, is_auth_i, secs_i) in enumerate(rec):
                            if id_i == _current_identity:
                                primary_idx = i
                                break

                    (bbox, identity, dist_raw, is_auth_raw, secs) = rec[primary_idx]
                    secondary_faces = [rec[i] for i in range(len(rec)) if i != primary_idx]

                    _last_detect_recog_s = secs
                    _smoothed_dist = ema(_smoothed_dist, float(dist_raw), DIST_SMOOTH_ALPHA)

                    x1, y1, x2, y2 = bbox
                    _smoothed_bbox = ema_bbox(_smoothed_bbox, (x1, y1, x2, y2), BBOX_SMOOTH_ALPHA)
                    x1, y1, x2, y2 = map(lambda v: int(round(v)), _smoothed_bbox)

                    # --------- AUTH STATE (PRIMARY FACE ONLY) ----------
                    raw_ok = bool(is_auth_raw) and (identity is not None) and (identity != "Unknown")

                    if _current_identity == identity and _current_identity is not None:
                        is_authorized = (
                            raw_ok
                            and _smoothed_dist is not None
                            and _smoothed_dist <= REVOKE_DIST_THRESH
                        )
                    else:
                        is_authorized = (
                            raw_ok
                            and _smoothed_dist is not None
                            and _smoothed_dist <= ACCEPT_DIST_THRESH
                        )

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

                    desired_color = (0, 200, 0) if is_authorized else (0, 0, 255)
                    if is_authorized:
                        desired_label = f"{identity}"
                    else:
                        desired_label = (
                            f"NOT AUTHORIZED d={_smoothed_dist:.2f}"
                            if _smoothed_dist is not None
                            else "NOT AUTHORIZED"
                        )

                    # Hold logic for PRIMARY box only
                    if (desired_color != _last_draw_color) or (desired_label != _last_draw_label):
                        if _hold_counter <= 0:
                            _last_draw_color = desired_color
                            _last_draw_label = desired_label
                            _hold_counter = COLOR_HOLD_FRAMES
                    else:
                        _hold_counter = max(0, _hold_counter - 1)

                    # Draw PRIMARY face with smoothed bbox (thick, green/red)
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), _last_draw_color, DRAW_THICKNESS)
                    (tw, th), _ = cv2.getTextSize(_last_draw_label, FONT, 0.6, 2)
                    cv2.rectangle(
                        frame_disp,
                        (x1, y2 + 4),
                        (x1 + tw + 6, y2 + th + 10),
                        _last_draw_color,
                        -1,
                    )
                    cv2.putText(
                        frame_disp,
                        _last_draw_label,
                        (x1 + 3, y2 + th + 3),
                        FONT,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # ---------- ATTENDANCE EVENT (PRIMARY ONLY) ----------
                    _update_streak_and_maybe_log(identity, is_authorized, _smoothed_dist)

                    # --------- INIT TRACKER ON PRIMARY FACE ----------
                    _tracker = _create_tracker(TRACKER_TYPE)
                    if _tracker is not None:
                        _tracker_bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                        try:
                            _tracker.init(frame_disp, _tracker_bbox)
                        except Exception:
                            _tracker = None
                            _tracker_bbox = None
                    else:
                        _tracker_bbox = None

                    # --------- SECONDARY FACES ----------
                    _last_secondary_overlays = []
                    seen_ids = set()

                    if secondary_faces:
                        for (bbox2, id2, dist2, is_auth2, _secs2) in secondary_faces:
                            if id2 is None:
                                continue

                            x1b, y1b, x2b, y2b = bbox2

                            is_auth2_final = (
                                is_auth2
                                and id2 != "Unknown"
                                and dist2 is not None
                                and dist2 <= ACCEPT_DIST_THRESH
                            )

                            color2 = (0, 200, 0) if is_auth2_final else (0, 0, 255)
                            if is_auth2_final:
                                label2 = id2
                            else:
                                label2 = (
                                    f"NOT AUTH d={dist2:.2f}"
                                    if dist2 is not None
                                    else "NOT AUTHORIZED"
                                )

                            cv2.rectangle(frame_disp, (x1b, y1b), (x2b, y2b), color2, DRAW_THICKNESS)
                            (tw2, th2), _ = cv2.getTextSize(label2, FONT, 0.6, 2)
                            cv2.rectangle(
                                frame_disp,
                                (x1b, y2b + 4),
                                (x1b + tw2 + 6, y2b + th2 + 10),
                                color2,
                                -1,
                            )
                            cv2.putText(
                                frame_disp,
                                label2,
                                (x1b + 3, y2b + th2 + 3),
                                FONT,
                                0.6,
                                (255, 255, 255),
                                2,
                            )

                            _last_secondary_overlays.append((x1b, y1b, x2b, y2b, label2, color2))
                            _update_streak_and_maybe_log(id2, is_auth2_final, dist2)
                            seen_ids.add(id2)

                    for k in list(_per_id_streak.keys()):
                        if k not in seen_ids and k != _current_identity:
                            _per_id_streak[k] = 0

            else:
                # tracker update
                if _tracker is not None and _tracker_bbox is not None:
                    ok_tr, tbox = _tracker.update(frame_disp)
                    if ok_tr:
                        x, y, w, h = map(int, tbox)
                        _smoothed_bbox = ema_bbox(
                            _smoothed_bbox,
                            (x, y, x + w, y + h),
                            BBOX_SMOOTH_ALPHA
                        )
                        sx1, sy1, sx2, sy2 = map(lambda v: int(round(v)), _smoothed_bbox)
                        cv2.rectangle(
                            frame_disp,
                            (sx1, sy1),
                            (sx2, sy2),
                            _last_draw_color,
                            DRAW_THICKNESS
                        )
                        (tw, th), _ = cv2.getTextSize(_last_draw_label, FONT, 0.6, 2)
                        cv2.rectangle(
                            frame_disp,
                            (sx1, sy2 + 4),
                            (sx1 + tw + 6, sy2 + th + 10),
                            _last_draw_color,
                            -1,
                        )
                        cv2.putText(
                            frame_disp,
                            _last_draw_label,
                            (sx1 + 3, sy2 + th + 3),
                            FONT,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        _hold_counter = max(0, _hold_counter - 1)

                        # ALSO draw last secondary overlays
                        for (x1b, y1b, x2b, y2b, label2, color2) in _last_secondary_overlays:
                            cv2.rectangle(frame_disp, (x1b, y1b), (x2b, y2b), color2, DRAW_THICKNESS)
                            (tw2, th2), _ = cv2.getTextSize(label2, FONT, 0.6, 2)
                            cv2.rectangle(
                                frame_disp,
                                (x1b, y2b + 4),
                                (x1b + tw2 + 6, y2b + th2 + 10),
                                color2,
                                -1,
                            )
                            cv2.putText(
                                frame_disp,
                                label2,
                                (x1b + 3, y2b + th2 + 3),
                                FONT,
                                0.6,
                                (255, 255, 255),
                                2,
                            )
                    else:
                        _tracker = None
                        _tracker_bbox = None
                        _last_secondary_overlays = []

            # ---------- overlays (countdown / banners / etc.) ----------
            now_wall = time.time()
            with _ui_lock:
                if _countdown_active:
                    secs_left = int(max(0, round(_countdown_end_ts - now_wall)))
                    if secs_left > 0:
                        overlay = frame_disp.copy()
                        cv2.rectangle(
                            overlay,
                            (0, 0),
                            (overlay.shape[1], overlay.shape[0]),
                            (0, 0, 0),
                            -1
                        )
                        frame_disp = cv2.addWeighted(overlay, 0.35, frame_disp, 0.65, 0)
                        label = f"{_countdown_label}"
                        (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
                        cx = frame_disp.shape[1] // 2 - tw // 2
                        cy = frame_disp.shape[0] // 2 - 80
                        cv2.putText(
                            frame_disp,
                            label,
                            (cx, cy),
                            FONT,
                            0.9,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )
                        num = str(secs_left)
                        (tw2, th2), _ = cv2.getTextSize(num, FONT, 3.0, 8)
                        cx2 = frame_disp.shape[1] // 2 - tw2 // 2
                        cy2 = frame_disp.shape[0] // 2 + th2 // 2
                        cv2.putText(
                            frame_disp,
                            num,
                            (cx2, cy2),
                            FONT,
                            3.0,
                            (0, 255, 0),
                            8,
                            cv2.LINE_AA
                        )
                    else:
                        _countdown_active = False

                if _flash_capture_until > now_wall:
                    cv2.rectangle(
                        frame_disp,
                        (4, 4),
                        (frame_disp.shape[1] - 4, frame_disp.shape[0] - 4),
                        (0, 255, 0),
                        6
                    )
                    text = "Capturing..."
                    (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 3)
                    x = frame_disp.shape[1] // 2 - tw // 2
                    cv2.rectangle(
                        frame_disp,
                        (x - 10, 45 - th - 10),
                        (x + tw + 10, 45 + 10),
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        frame_disp,
                        text,
                        (x, 45),
                        FONT,
                        0.9,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA
                    )

                if _name_prompt_banner_until > now_wall:
                    msg = "اكتب اسمك في نافذة الأوامر ثم اضغط Enter..."
                    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
                    cv2.rectangle(
                        frame_disp,
                        (10, 10),
                        (10 + tw + 14, 10 + th + 20),
                        (60, 60, 220),
                        -1
                    )
                    cv2.putText(
                        frame_disp,
                        msg,
                        (17, 10 + th + 10),
                        FONT,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                if _banner_until > now_wall:
                    msg = _banner_text
                    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
                    x = frame_disp.shape[1] // 2 - tw // 2
                    y = 40
                    cv2.rectangle(
                        frame_disp,
                        (x - 10, y - th - 10),
                        (x + tw + 10, y + 10),
                        (0, 200, 0),
                        -1
                    )
                    cv2.putText(
                        frame_disp,
                        msg,
                        (x, y),
                        FONT,
                        0.8,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA
                    )

            # stats
            y = 24

            def put(t):
                nonlocal y
                cv2.putText(
                    frame_disp,
                    t,
                    (10, y),
                    FONT,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame_disp,
                    t,
                    (10, y),
                    FONT,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                y += 22

            put(f"Detect+Recognize: {_last_detect_recog_s:.3f} s")
            put(f"Auth latency: {_last_auth_latency_s:.3f} s")
            put(f"FPS: {fps:.1f}")
            put(f"Heavy cadence: ~{_detect_every_n} frames")
            cv2.putText(
                frame_disp,
                "Press 'g' to enroll",
                (10, frame_disp.shape[0] - 12),
                FONT,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Upscale ONLY for display; processing still uses small frame_disp
            display = frame_disp
            if UI_SCALE != 1.0:
                display = cv2.resize(
                    frame_disp, None,
                    fx=UI_SCALE,
                    fy=UI_SCALE,
                    interpolation=cv2.INTER_LINEAR
                )

            cv2.imshow(win_name, display)

            # -------- Keyboard handling --------
            key = cv2.waitKey(1) & 0xFF

            # ---- Decide if there is an enrollment trigger (keyboard or API) ----
            trigger_source = None

            # 1) Keyboard 'g'
            if key == ord('g'):
                trigger_source = "keyboard"

            # 2) API trigger
            if ENROLL_REQUESTED_FROM_API:
                trigger_source = "api"
                ENROLL_REQUESTED_FROM_API = False  # consume the request

            if trigger_source is not None and not _capture_in_progress:
                allowed, reason = enroll_allowed_now()
                if allowed:
                    if capt.start_capture():
                        _last_capture_trigger_ts = time.time()
                        _last_capture_frame_idx = frame_idx

                        if trigger_source == "keyboard":
                            print("[MANUAL] 'g' → capture started.")
                            speak("تم الضغط على جي. سيتم الآن تسجيل مستخدم جديد.")
                        else:
                            print("[API] HR requested enrollment → capture started.")
                            speak("تم طلب تسجيل مستخدم جديد من خلال لوحة الموارد البشرية.")
                    else:
                        print(f"[{trigger_source.upper()}] worker busy.")
                        speak("لا يمكن بدء التسجيل الآن. عملية سابقة ما زالت قيد التنفيذ.")
                else:
                    print(f"[{trigger_source.upper()}] capture blocked:", reason)
                    if "face too similar" in reason:
                        speak("لا يمكن إنشاء مستخدم جديد الآن. الوجه مشابه جداً لِوجه موجود.")
                    elif "recent authorization" in reason:
                        speak("تم التعرف على هذا الوجه منذ قليل. انتظر قليلاً قبل محاولة التسجيل مرة أخرى.")
                    elif "currently authenticated" in reason:
                        speak("هذا الوجه مسجل بالفعل كمستخدم معتمد.")
                    elif "cooldown" in reason:
                        speak("من فضلك انتظر قليلاً قبل محاولة التسجيل مرة أخرى.")
                    else:
                        speak("لا يمكن بدء التسجيل في الوقت الحالي.")

            # -------- Quit key --------
            if key == ord('q'):
                break

            # -------- Finalize any pending enrollments + reload DB if needed --------
            finalize_all(model_pair)
            with _request_db_reload_lock:
                if _request_db_reload:
                    print("[DB] reloading templates...")
                    DB_TEMPLATES = db_templates_dict()
                    _request_db_reload = False

    finally:
        try:
            recog.shutdown()
        except Exception:
            pass
        # REMOVE this block (no _tts anymore):
        # try:
        #     _tts.shutdown()
        # except Exception:
        #     pass
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass
    main()