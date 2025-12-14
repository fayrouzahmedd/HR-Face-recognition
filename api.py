# api.py
"""
Simple FastAPI server for the Face Recognition system.

- On startup: starts hr_fr_cam2.main() in a background thread.
- POST /enroll              → Start enrollment (same as pressing 'g').
- GET  /health              → API health check.
- GET  /system/status       → Face system running? capture in progress?
- GET  /enroll/status       → Is enrollment allowed now?
- GET  /camera/status       → Current camera source (IP/local).
- GET  /version             → API + model info.
- GET  /attendance          → Read attendance records from SQLite.
- GET  /persons             → List all persons + templates count.
- GET  /persons/{person_id} → Person details + attendance history.
- GET  /persons/{person_id}/templates → Template metadata (no embeddings).
- GET  /new-users           → Enrollment audit log.

- PUT    /persons/{person_id}        → Rename a person (also updates attendance.name).
- PATCH  /attendance/{attendance_id} → Edit an attendance record.
- DELETE /persons/{person_id}        → Delete a person (templates/images cascade).
- DELETE /attendance/{attendance_id} → Delete an attendance record.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sqlite3

# This imports your big face-recognition script (hr_fr_cam2.py)
import hr_fr_cam2


# ================== FASTAPI APP ==================

app = FastAPI(
    title="Face Enrollment API",
    version="1.2.0",
    description="API exposing enrollment trigger, status endpoints, and database admin endpoints.",
)


# ================== DB HELPERS ==================

def _db_connect_api():
    """
    Separate DB connector for the API.

    IMPORTANT:
    - SQLite foreign keys are OFF by default per connection → must enable every time.
    - WAL mode improves concurrency for reads while your camera system writes.
    """
    db_path = str(hr_fr_cam2.AUTH_DB_PATH)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


# ================== SCHEMA MODELS ==================

class EnrollResponse(BaseModel):
    success: bool
    message: str


class AttendanceRecord(BaseModel):
    id: int
    name: str
    status: str          # 'Arrival' or 'Departure'
    ts_iso: str          # ISO timestamp
    distance: Optional[float] = None


class PersonUpdate(BaseModel):
    new_name: str


class AttendanceUpdate(BaseModel):
    # Partial update allowed
    name: Optional[str] = None
    status: Optional[str] = None          # 'Arrival' or 'Departure'
    ts_iso: Optional[str] = None          # ISO string e.g. 2025-12-14T12:30:10
    distance: Optional[float] = None


class GenericResponse(BaseModel):
    success: bool
    message: str


# ================== STARTUP ==================

@app.on_event("startup")
def startup_event():
    """
    Runs ONCE when FastAPI starts:
    - launches the face system in a background thread.
    """
    print("[API] Startup: launching face recognition system...")
    hr_fr_cam2.start_system_in_background()


# ================== POST: ENROLLMENT TRIGGER ==================

@app.post("/enroll", response_model=EnrollResponse)
def enroll_new_user():
    """
    Trigger a new-user enrollment, same as pressing 'g' in the UI.
    """
    ok, reason = hr_fr_cam2.request_enroll_from_api()
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    return EnrollResponse(
        success=True,
        message=(
            "Enrollment started. Ask the employee to stand in front of the camera "
            "and follow the voice instructions."
        ),
    )


# ================== GET: STATUS ENDPOINTS ==================

@app.get("/health")
def health():
    return {"status": "ok", "service": "face-enrollment-api"}


@app.get("/system/status")
def system_status():
    face_thread_alive = (
        getattr(hr_fr_cam2, "_face_thread", None) is not None
        and getattr(hr_fr_cam2._face_thread, "is_alive", lambda: False)()
    )
    return {
        "face_system_running": face_thread_alive,
        "capture_in_progress": getattr(hr_fr_cam2, "_capture_in_progress", False),
        "enroll_requested": getattr(hr_fr_cam2, "ENROLL_REQUESTED_FROM_API", False),
    }


@app.get("/enroll/status")
def enroll_status():
    allowed, reason = hr_fr_cam2.enroll_allowed_now()
    return {"allowed": allowed, "reason": reason}


@app.get("/camera/status")
def camera_status():
    use_ip = getattr(hr_fr_cam2, "USE_IP_CAMERA", False)
    if use_ip:
        source = getattr(hr_fr_cam2, "IP_CAM_URL", "unknown")
    else:
        cam_index = getattr(hr_fr_cam2, "CAM_INDEX", 0)
        source = f"local_cam_{cam_index}"

    return {"use_ip_camera": use_ip, "camera_source": source}


@app.get("/version")
def version():
    model_name = getattr(hr_fr_cam2, "MODEL_NAME", "unknown")
    fast_mode = getattr(hr_fr_cam2, "FAST_MODE", None)
    return {"api_version": app.version, "model_name": model_name, "fast_mode": fast_mode}


# ================== GET: ATTENDANCE VIEW ==================

@app.get("/attendance", response_model=List[AttendanceRecord])
def get_attendance(
    name: Optional[str] = None,
    person_id: Optional[int] = None,
    date: Optional[str] = None,   # 'YYYY-MM-DD'
    limit: int = 100,
):
    """
    Read attendance rows from SQLite (latest first).

    Filters (all optional):
    - name: exact match on attendance.name
    - person_id: match on attendance.person_id
    - date: YYYY-MM-DD match on ts_iso day (substr(ts_iso,1,10))

    Rules:
    - Do NOT send both name and person_id (ambiguous) -> 400
    """

    # safety
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000

    if name is not None:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name cannot be empty")

    if date is not None:
        date = date.strip()
        if len(date) != 10:
            raise HTTPException(status_code=400, detail="date must be in format YYYY-MM-DD")

    # avoid ambiguous filters
    if name and person_id is not None:
        raise HTTPException(status_code=400, detail="Provide either name OR person_id, not both")

    sql = """
        SELECT id, name, status, ts_iso, distance
        FROM attendance
        WHERE 1=1
    """
    params = []

    if name:
        sql += " AND name = ?"
        params.append(name)

    if person_id is not None:
        sql += " AND person_id = ?"
        params.append(int(person_id))

    if date:
        sql += " AND substr(ts_iso, 1, 10) = ?"
        params.append(date)

    sql += " ORDER BY ts_unix DESC LIMIT ?"
    params.append(limit)

    try:
        con = _db_connect_api()
        rows = con.execute(sql, params).fetchall()
        con.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return [
        AttendanceRecord(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            ts_iso=row["ts_iso"],
            distance=row["distance"],
        )
        for row in rows
    ]


# ================== GET: PERSONS ==================

@app.get("/persons")
def list_persons():
    """
    List all enrolled persons with templates count.
    """
    try:
        con = _db_connect_api()
        rows = con.execute("""
            SELECT
                p.id,
                p.name,
                p.created_at,
                COUNT(t.id) AS templates_count
            FROM persons p
            LEFT JOIN templates t ON t.person_id = p.id
            GROUP BY p.id
            ORDER BY p.name ASC
        """).fetchall()
        con.close()

        return [
            {
                "person_id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "templates_count": row["templates_count"],
            }
            for row in rows
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.get("/persons/{person_id}")
def get_person_details(person_id: int):
    """
    Person details + templates count + attendance history.
    """
    try:
        con = _db_connect_api()

        person = con.execute(
            "SELECT id, name, created_at FROM persons WHERE id=?",
            (person_id,)
        ).fetchone()

        if not person:
            con.close()
            raise HTTPException(status_code=404, detail="person not found")

        templates_count = con.execute(
            "SELECT COUNT(*) FROM templates WHERE person_id=?",
            (person_id,)
        ).fetchone()[0]

        attendance = con.execute("""
            SELECT id, status, ts_iso, distance
            FROM attendance
            WHERE person_id=?
            ORDER BY ts_unix DESC
        """, (person_id,)).fetchall()

        con.close()

        return {
            "person_id": person["id"],
            "name": person["name"],
            "created_at": person["created_at"],
            "templates_count": templates_count,
            "attendance": [
                {
                    "attendance_id": row["id"],
                    "status": row["status"],
                    "ts_iso": row["ts_iso"],
                    "distance": row["distance"],
                }
                for row in attendance
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.get("/persons/{person_id}/templates")
def get_person_templates(person_id: int):
    """
    Template metadata only (no embeddings).
    """
    try:
        con = _db_connect_api()

        exists = con.execute("SELECT 1 FROM persons WHERE id=?", (person_id,)).fetchone()
        if not exists:
            con.close()
            raise HTTPException(status_code=404, detail="person not found")

        rows = con.execute("""
            SELECT id, created_at
            FROM templates
            WHERE person_id=?
            ORDER BY created_at ASC
        """, (person_id,)).fetchall()

        con.close()

        return {
            "person_id": person_id,
            "templates": [
                {"template_id": row["id"], "created_at": row["created_at"]}
                for row in rows
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.get("/new-users")
def get_new_users(limit: int = 100):
    """
    Enrollment audit log (new_users_log).
    """
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000

    try:
        con = _db_connect_api()
        rows = con.execute("""
            SELECT id, person_id, name, ts_iso
            FROM new_users_log
            ORDER BY ts_unix DESC
            LIMIT ?
        """, (limit,)).fetchall()
        con.close()

        return [
            {
                "log_id": row["id"],
                "person_id": row["person_id"],
                "name": row["name"],
                "ts_iso": row["ts_iso"],
            }
            for row in rows
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# ================== PUT/PATCH: UPDATE ==================

@app.put("/persons/{person_id}", response_model=GenericResponse)
def update_person_name(person_id: int, payload: PersonUpdate):
    new_name = payload.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="new_name cannot be empty")

    try:
        con = _db_connect_api()
        cur = con.cursor()

        row = cur.execute("SELECT id, name FROM persons WHERE id=?", (person_id,)).fetchone()
        if not row:
            con.close()
            raise HTTPException(status_code=404, detail="person not found")

        old_name = row["name"]

        exists = cur.execute("SELECT 1 FROM persons WHERE name=? LIMIT 1", (new_name,)).fetchone()
        if exists:
            con.close()
            raise HTTPException(status_code=409, detail="name already exists")

        cur.execute("UPDATE persons SET name=? WHERE id=?", (new_name, person_id))
        cur.execute("UPDATE attendance SET name=? WHERE name=?", (new_name, old_name))

        con.commit()
        con.close()

        # optional: refresh templates in running face system
        try:
            with hr_fr_cam2._request_db_reload_lock:
                hr_fr_cam2._request_db_reload = True
        except Exception:
            pass

        return GenericResponse(success=True, message=f"Renamed '{old_name}' -> '{new_name}'")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.patch("/attendance/{attendance_id}", response_model=GenericResponse)
def update_attendance(attendance_id: int, payload: AttendanceUpdate):
    allowed_status = {"Arrival", "Departure"}
    updates = []
    params = []

    if payload.name is not None:
        nm = payload.name.strip()
        if not nm:
            raise HTTPException(status_code=400, detail="name cannot be empty")
        updates.append("name=?")
        params.append(nm)

    if payload.status is not None:
        st = payload.status.strip()
        if st not in allowed_status:
            raise HTTPException(status_code=400, detail="status must be Arrival or Departure")
        updates.append("status=?")
        params.append(st)

    if payload.ts_iso is not None:
        ts = payload.ts_iso.strip()
        if not ts:
            raise HTTPException(status_code=400, detail="ts_iso cannot be empty")
        updates.append("ts_iso=?")
        params.append(ts)

    if payload.distance is not None:
        updates.append("distance=?")
        params.append(float(payload.distance))

    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    params.append(attendance_id)

    try:
        con = _db_connect_api()
        cur = con.cursor()

        row = cur.execute("SELECT id FROM attendance WHERE id=?", (attendance_id,)).fetchone()
        if not row:
            con.close()
            raise HTTPException(status_code=404, detail="attendance record not found")

        sql = f"UPDATE attendance SET {', '.join(updates)} WHERE id=?"
        cur.execute(sql, params)

        con.commit()
        con.close()
        return GenericResponse(success=True, message="Attendance updated successfully")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# ================== DELETE ==================

@app.delete("/persons/{person_id}", response_model=GenericResponse)
def delete_person(person_id: int):
    """
    Deletes a person:
    - templates/images cascade
    - attendance.person_id becomes NULL (history preserved)
    """
    try:
        con = _db_connect_api()
        cur = con.cursor()

        row = cur.execute("SELECT id, name FROM persons WHERE id=?", (person_id,)).fetchone()
        if not row:
            con.close()
            raise HTTPException(status_code=404, detail="person not found")

        name = row["name"]

        cur.execute("DELETE FROM persons WHERE id=?", (person_id,))
        con.commit()
        con.close()

        # optional: refresh templates in running face system
        try:
            with hr_fr_cam2._request_db_reload_lock:
                hr_fr_cam2._request_db_reload = True
        except Exception:
            pass

        return GenericResponse(success=True, message=f"Deleted person '{name}' (id={person_id})")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.delete("/attendance/person/{person_id}", response_model=GenericResponse)
def delete_attendance_by_person_and_date(person_id: int, date: str):
    """
    Delete attendance records ONLY if:
    - person_id matches
    - date (YYYY-MM-DD) matches ts_iso day

    Example:
    DELETE /attendance/person/13?date=2025-12-10
    """
    if not date or len(date) != 10:
        raise HTTPException(
            status_code=400,
            detail="date query parameter is required in format YYYY-MM-DD"
        )

    try:
        con = _db_connect_api()
        cur = con.cursor()

        # Check matching records
        rows = cur.execute("""
            SELECT id, ts_iso
            FROM attendance
            WHERE person_id = ?
              AND substr(ts_iso, 1, 10) = ?
        """, (person_id, date)).fetchall()

        if not rows:
            con.close()
            raise HTTPException(
                status_code=404,
                detail="no attendance records found for given person_id and date"
            )

        # Delete all matching rows (Arrival / Departure etc.)
        cur.execute("""
            DELETE FROM attendance
            WHERE person_id = ?
              AND substr(ts_iso, 1, 10) = ?
        """, (person_id, date))

        deleted_count = cur.rowcount

        con.commit()
        con.close()

        return GenericResponse(
            success=True,
            message=f"Deleted {deleted_count} attendance record(s) for person_id={person_id} on {date}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

# ================== RUN (for reference) ==================
# uvicorn api:app --reload --port 8000