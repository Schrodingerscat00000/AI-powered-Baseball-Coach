import os, io, csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
import openai, pinecone, pandas as pd
from pinecone import Pinecone, ServerlessSpec
from typing import List, Optional, Dict, Any
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from pymongo import MongoClient
from datetime import timedelta
from fastapi import Query
import base64
import re
import tempfile
from dateutil import parser as dateparser   # pip install python-dateutil
import requests
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
import logging, traceback
logger = logging.getLogger(__name__)


#from models import DailyLog  # your Pydantic model
import json
from bson import ObjectId  # ← for converting string → ObjectId

load_dotenv()
# Init services
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create Pinecone client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# Optional: create index if not exists
index_name = "baseball"  # replace with your actual index name

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # match embedding model's output
        metric='cosine',  # or 'euclidean'
        spec=ServerlessSpec(
            cloud='aws',         # or 'gcp'
            region='us-east-1'   # your Pinecone region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# ── Set up your Mongo client ─────────────────────────────────────────────
mongo = MongoClient(
    "mongodb+srv://test-user:test-user@cluster0.kdy82ie.mongodb.net/"
    "?retryWrites=true&w=majority&appName=Cluster0"
)
db = mongo["gudbaseball"]
dailylogs = db["dailylogs"]
chat_messages = db["chat_messages"]


app = FastAPI()

@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"message": "hello"}

class DailyLog(BaseModel):
    userId: str
    date: datetime
    visualization: Optional[Dict[str, Any]]           = None
    dailyWellnessQuestionnaire: Optional[Dict[str, Any]] = None
    throwingJournal: Optional[Dict[str, Any]]         = None
    armCare: Optional[Dict[str, Any]]                  = None
    Lifting: Optional[Dict[str, Any]]                  = None
    hittingJournal: Optional[Dict[str, Any]]           = None
    postPerformance: Optional[Dict[str, Any]]          = None

    class Config:
        extra = "allow"           # ignore any other keys

class ChatRequest(BaseModel):
  userId: str
  message: str

class FilterReq(BaseModel):
    userId: str
    start: datetime = Field(..., alias="startDate")
    end:   datetime = Field(..., alias="endDate")
    class Config:
        allow_population_by_field_name = True

class IntentReq(BaseModel):
    intent: str
    user_id: str
    start: Optional[str] = None
    end: Optional[str] = None
    sections: Optional[list] = None
    format: Optional[str] = "csv"
    raw_text: Optional[str] = None


# ─── Utilities ──────────────────────────────────────────────────────────────
def humanize_section(name: str, val: Any) -> str:
    if not val:
        return ""
    # Special handling for nutrition
    if name == "nutrition" and isinstance(val, dict):
        parts = []
        if "nutritionScore" in val:
            parts.append(f"nutrition score: {val['nutritionScore']}/10")
        if "proteinInGram" in val:
            parts.append(f"protein intake: {val['proteinInGram']}g")
        if "caloricScore" in val:
            parts.append(f"caloric score: {val['caloricScore']}/10")
        if "consumedImpedingSubstances" in val:
            parts.append(f"impeding substances: {val['consumedImpedingSubstances']}")
        return "; ".join(parts)

    # Generic handling for other dict sections
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            label = k.replace('_', ' ')
            if isinstance(v, list):
                parts.append(f"{label}: {' and '.join(map(str, v))}")
            else:
                parts.append(f"{label}: {v}")
        return f"{name}: " + "; ".join(parts)

    # Fallback for primitives
    return f"{name}: {val}"



# ─── Embed Endpoint ─────────────────────────────────────────────────────────
@app.post("/embed")
def embed(log: DailyLog):
    # 1) Upsert into Mongo
    filter_ = {"userId": log.userId, "date": log.date}
    update_ = {"$set": log.dict(exclude_none=True)}
    entry = dailylogs.find_one_and_update(
        filter_, update_, upsert=True, return_document=ReturnDocument.AFTER
    )

    # 2) Humanize & concatenate
    sections = [
        "visualization","dailyWellnessQuestionnaire","throwingJournal",
        "armCare","Lifting","hittingJournal","postPerformance","nutrition"
    ]
    texts = [humanize_section(sec, entry.get(sec)) for sec in sections]
    text = "\n".join([t for t in texts if t]) or "No data"

    # 3) Embed
    resp = openai.embeddings.create(input=[text], model="text-embedding-3-small")
    emb = resp.data[0].embedding

    # 4) Metadata (primitives or JSON string)
    metadata = {"userId": entry["userId"], "date": entry["date"].isoformat()}
    for sec in sections:
        v = entry.get(sec)
        if v is None:
            continue
        if isinstance(v, (str,bool,int,float)):
            metadata[sec] = v
        else:
            metadata[sec] = json.dumps(v, default=str)

    # 5) Upsert to Pinecone
    index.upsert([(f"{entry['userId']}:{entry['date'].isoformat()}", emb, metadata)])
    return {"status":"ok"}


def make_csv_string(req: FilterReq) -> (str, str):
    # Exactly your existing export_csv logic, but returning CSV text & filename
    raw_uid = req.userId.strip()
    try:
        user_oid = ObjectId(raw_uid)
    except:
        raise HTTPException(400, "`userId` is not a valid ObjectId")

    query = {"userId": user_oid, "date": {"$gte": req.start, "$lte": req.end}}
    docs = list(dailylogs.find(query).sort("date", 1))

    sections = [
        "visualization","dailyWellnessQuestionnaire","throwingJournal",
        "armCare","Lifting","hittingJournal","postPerformance","nutrition"
    ]
    rows = []
    for d in docs:
        dt_iso = d["date"].isoformat()
        row = {"date": dt_iso}
        for sec in sections:
            v = d.get(sec)
            if v is None:
                row[sec] = ""
            elif isinstance(v, (str,int,float,bool)):
                row[sec] = v
            else:
                row[sec] = json.dumps(v, default=str)
        rows.append(row)

    # Build CSV
    header = ["date"] + sections
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    csv_text = buf.getvalue()
    filename = f"{raw_uid}_logs_{req.start.date()}_to_{req.end.date()}.csv"
    return csv_text, filename


def user_has_permission(user_id: str) -> bool:
    """
    Minimal permissive permission check:
    - Returns True if we find any record in dailylogs or chat_messages for that userId.
    - Replace with strict auth (JWT/session check + user DB) in production.
    """
    try:
        # prefer using your existing collections if available
        if "dailylogs" in globals():
            if dailylogs.find_one({"userId": ObjectId(user_id)}) if len(user_id) == 24 else dailylogs.find_one({"userId": user_id}):
                return True
        if "chat_messages" in globals():
            if chat_messages.find_one({"userId": user_id}):
                return True
    except Exception:
        # if checking by ObjectId failed, try plain string match
        try:
            if "dailylogs" in globals() and dailylogs.find_one({"userId": user_id}):
                return True
        except Exception:
            pass
    # Fallback: allow (or change to False to be strict)
    return True


def upload_and_sign_to_storage(filename: str, data: str, content_type: str = "text/csv") -> str:
    """
    DEV fallback: write file to /tmp and return file:// path.
    Replace with S3/MinIO uploader + presigned URL for production.
    """
    try:
        tmpdir = tempfile.gettempdir()
        safe_name = filename.replace("/", "_")
        path = os.path.join(tmpdir, safe_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        # Return a file URL for local testing. Your front-end can handle it in dev.
        return f"file://{path}"
    except Exception:
        traceback.print_exc()
        raise


def generate_and_notify(job: dict):
    """
    Background job runner. Job dict keys: user_id, start, end, sections, format (optional).
    - Generates CSV using make_csv_string()
    - Uploads with upload_and_sign_to_storage()
    - Optionally records the export in an 'exports' collection and tries to email user (no-op here).
    If job.get('sync') is True, returns the download URL (useful for synchronous calls).
    """
    try:
        user_id = job.get("user_id") or job.get("userId") or job.get("user")
        start_raw = job.get("start")
        end_raw = job.get("end")
        fmt = job.get("format", "csv")

        start_dt = dateparser.parse(start_raw) if start_raw else None
        end_dt = dateparser.parse(end_raw) if end_raw else None

        if not start_dt:
            start_dt = datetime.utcnow() - timedelta(days=7)
        if not end_dt:
            end_dt = datetime.utcnow()

        # Build FilterReq (respect alias fields)
        filt = FilterReq(userId=user_id, startDate=start_dt, endDate=end_dt)
        csv_text, filename = make_csv_string(filt)

        # If user asked for xlsx/json conversion you should do it here. For now only csv.
        url = upload_and_sign_to_storage(filename, csv_text, content_type="text/csv")

        # Try to store metadata in an 'exports' collection if you have DB
        try:
            if "dailylogs" in globals():
                db = dailylogs.database if hasattr(dailylogs, "database") else None
                if db is not None:
                    exports_coll = db.get_collection("exports")
                    exports_coll.insert_one({
                        "userId": user_id,
                        "filename": filename,
                        "download_url": url,
                        "created_at": datetime.utcnow(),
                        "status": "ready"
                    })
        except Exception:
            # non-fatal
            traceback.print_exc()

        # Optionally, send an email/notification. Left as a no-op in this stub.
        # If job has sync True, return URL for caller.
        if job.get("sync"):
            return url
        return None
    except Exception:
        traceback.print_exc()
        return None

# -------------------- Download intent extractor --------------------
SECTIONS = {
    "visualization","dailywellnessquestionnaire","throwingjournal",
    "armcare","lifting","hittingjournal","postperformance","nutrition",
    "logs","data","history","wellness"
}
FORMAT_ALIASES = {"csv": "csv", "excel": "xlsx", "xlsx": "xlsx", "json": "json"}

def parse_dates_from_text(text: str):
    now = datetime.utcnow()
    text = text.lower()

    # presets
    if re.search(r"\b(last|past)\s*(30|thirty)\b", text):
        return now - timedelta(days=30), now, "last_30"
    if re.search(r"\b(last|past)\s*(7|seven)\b", text):
        return now - timedelta(days=7), now, "last_7"
    if re.search(r"\b(all time|all-time|everything|all data)\b", text):
        return None, None, "all_time"

    m = re.search(r"last\s+(\d{1,3})\s+days?", text)
    if m:
        days = int(m.group(1))
        return now - timedelta(days=days), now, f"last_{days}"

    # "from X to Y"
    m = re.search(r"from\s+([^,]+?)\s+(to|-)\s+([^,]+)", text)
    if m:
        try:
            start = dateparser.parse(m.group(1), fuzzy=True)
            end = dateparser.parse(m.group(3), fuzzy=True)
            return start, end, None
        except Exception:
            pass

    m = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})", text)
    if m:
        try:
            start = dateparser.parse(m.group(1))
            end = dateparser.parse(m.group(2))
            return start, end, None
        except Exception:
            pass

    # since <date>
    m = re.search(r"\bsince\s+([^\n,]+)", text)
    if m:
        try:
            start = dateparser.parse(m.group(1), fuzzy=True)
            return start, now, None
        except Exception:
            pass

    # fallback to last 7 days
    return now - timedelta(days=7), now, "last_7"


def parse_sections(text: str):
    found = set()
    t = text.lower()
    for s in SECTIONS:
        if s in t:
            found.add(s)
    if re.search(r"\b(all|everything|all data|logs)\b", t) and not found:
        return list(SECTIONS)
    return list(found)


def parse_format(text: str):
    t = text.lower()
    for k in FORMAT_ALIASES:
        if k in t:
            return FORMAT_ALIASES[k]
    return "csv"


def detect_download_intent(text: str):
    t = text.lower()
    # trigger verbs or nouns
    if not (re.search(r"\b(download|export|send|give|get|retrieve|downloadable|send me)\b", t)
            or re.search(r"\b(data|logs|log|csv|report|history|records)\b", t)):
        return None

    start, end, preset = parse_dates_from_text(t)
    sections = parse_sections(t)
    fmt = parse_format(t)
    return {
        "intent": "download_data",
        "preset": preset,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "sections": sections,
        "format": fmt,
        "raw_text": text
    }


# ─── Chat Endpoint ──────────────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks):

    text = req.message.lower()

    # --------- improved download intent handling ----------
    intent = detect_download_intent(text)
    if intent:
        # fill user id
        intent["user_id"] = req.userId

        # If the user included no explicit date and no sections, ask to confirm (one quick clarifier)
        no_dates = (intent["preset"] is None and not intent["start"] and not intent["end"])
        no_sections = not intent["sections"]
        if no_dates and no_sections and "last" not in intent.get("raw_text",""):
            # ask a clarifying question rather than assuming
            return {
                "tag": "clarify_download",
                "reply": "I can export your logs. Do you want the last 7 days, last 30 days, or a custom date range?"
            }

        # send intent to backend /intent endpoint (same app or separate service)
        # build IntentReq and call internal handler directly
        try:
            ireq = IntentReq(**intent)            # intent already has user_id filled
            j = handle_intent(ireq, background_tasks)   # pass the FastAPI BackgroundTasks
        except Exception as e:
            logger.error("Export intent failed: %s", e)
            logger.error(traceback.format_exc())
            return {"tag":"error","reply":"Couldn't process export request. Try again or contact support."}

        if j.get("ok") and j.get("download_url"):
            return {"tag":"csv_download", "reply": f"I prepared your export — download here: {j['download_url']}"}
        else:
            return {"tag":"info", "reply": j.get("message", "Your export request is queued. We'll notify you when ready.")}


    # Thread last 2 messages
    history = list(chat_messages.find({"userId": req.userId})
                        .sort("timestamp",-1).limit(5))
    msg_history = [{"role":d["role"],"content":d["message"]} for d in reversed(history)]

    # Semantic search
    emb = openai.embeddings.create(
        input=[req.message], model="text-embedding-3-small"
    ).data[0].embedding
    qres = index.query(vector=emb, top_k=5, include_metadata=True,
                       filter={"userId": req.userId})

    # Build context
    context_parts = []
    for m in qres.matches:
        for k,v in m.metadata.items():
            if k in ("userId","date"): continue
            context_parts.append(f"{k}: {v}")
    if not context_parts:
        # fallback: last 3 raw logs
        docs = list(dailylogs.find({"userId":req.userId})
                           .sort("date",-1).limit(3))
        for doc in docs:
            dt = doc["date"]
            context_parts.append(f"{dt.strftime('%Y-%m-%d')}: "
                                 + ", ".join(k for k in doc.keys()
                                             if k not in ("_id","userId","date","timestamp")))
    context_block = ("Here’s your recent data:\n" + "\n".join(context_parts)
                     ) if context_parts else "No logs found. Please record data."

    # Prepend this thresholds summary to your system prompt
    thresholds_text = """
    Habit‑Tier Thresholds:
    • Visualization: Good ≥4/wk, Average 2–3, Needs Improvement 0–1  
    • Consistency:   Good ≥5/wk, Average 3–4, Needs Improvement 1–2, Not Serious 0  
    • Lifting (In‑Season Mar–Aug): Good ≥3/wk, Avg 2, Needs Focus 0–1  
    Off‑Season (Sep–Feb): Good ≥4, Avg 2–3, Needs Focus 1  
    • Recovery:      Good ≥4/wk, Average 2–3, Needs Focus 0–1  
    • Wellness:      Excellent ≥6, Good 4–5, Average 3, Needs Focus 0–2  
    • Nutrition Score: Excellent ≥8/10, Good 5–7, Average 3–4, Needs Improvement 0–2
    """

    # System prompt
    system_msgs = [
    {"role":"system","content": thresholds_text.strip()},
    {"role":"system","content":
     "You are a data-driven baseball coach AI for a single athlete."
     "Use ONLY the provided data context and the conversation to answer"
     "questions about training, performance, recovery, wellness, nutrition. "
     "When giving training advice, include: (1) A short summary of why, (2) a 3-step actionable plan the player can follow this week, and (3) one measurable checkpoint to track progress. Keep safety in mind and avoid medical advice—refer to a professional if injury signs appear."
     "You can also provide insights, summaries, and recommendations based on the athlete's logs and questions."
     "Refer to prior conversation for follow‑ups."},
    {"role":"system","content": context_block}
]

    # Assemble and call
    messages = system_msgs + msg_history + [{"role":"user","content":req.message}]
    comp = openai.chat.completions.create(
    model="gpt-4o-mini",   # or "gpt-4o" / your chosen model
    messages=messages,
    temperature=0.25,
    max_tokens=700
    )

    reply = comp.choices[0].message.content.strip()

    # Persist Q&A
    now = datetime.utcnow()
    chat_messages.insert_many([
        {"userId":req.userId,"role":"user","message":req.message,"timestamp":now},
        {"userId":req.userId,"role":"assistant","message":reply,"timestamp":now}
    ])
    print(f"[CHAT] Q: {req.message}\n      A: {reply}")
    return {"reply":reply}



@app.post("/intent")
def handle_intent(req: IntentReq, background_tasks: BackgroundTasks):
    if req.intent != "download_data":
        raise HTTPException(400, "unsupported intent")
    # permission check
    if not user_has_permission(req.user_id):
        return {"ok": False, "error": "permission_denied"}

    # parse datetimes if present
    start = dateparser.parse(req.start) if req.start else None
    end = dateparser.parse(req.end) if req.end else None

    # If small range -> generate immediately and upload to object storage and return signed URL.
    # Define is_small_range() as you wish (e.g., days <= 30)
    if start and end:
        days = (end - start).days
    else:
        days = 7  # default

    if days <= 30:
        # synchronous generation
        filt = FilterReq(userId=req.user_id, startDate=start or datetime.utcnow() - timedelta(days=7),
                         endDate=end or datetime.utcnow())
        csv_text, filename = make_csv_string(filt)
        # upload to storage and create signed URL. Replace with your S3/minio logic.
        signed_url = upload_and_sign_to_storage(filename, csv_text, content_type="text/csv")
        return {"ok": True, "download_url": signed_url}
    else:
        # enqueue background task to generate and email the user or provide status link
        job = {"user_id": req.user_id, "start": req.start, "end": req.end, "sections": req.sections, "format": req.format}
        background_tasks.add_task(generate_and_notify, job)
        return {"ok": True, "message": "Your export is queued. We'll email you when it's ready."}

@app.post("/export_csv")
def export_csv(req: FilterReq):
    # 1) Clean & validate userId
    raw_uid = req.userId.strip()
    if len(raw_uid) != 24:
        raise HTTPException(status_code=400, detail="`userId` must be a 24‑char hex string")
    try:
        user_oid = ObjectId(raw_uid)
    except Exception:
        raise HTTPException(status_code=400, detail="`userId` is not a valid ObjectId")

    # 2) Query MongoDB
    query = {
        "userId": user_oid,
        "date": {"$gte": req.start, "$lte": req.end}
    }
    raw_docs = list(dailylogs.find(query).sort("date", 1))

    # 3) Debug print
    print(f"DEBUG: fetched {len(raw_docs)} docs for user {raw_uid}")
    for doc in raw_docs:
        print(doc)

    # 4) Flatten rows
    sections = [
        "visualization",
        "dailyWellnessQuestionnaire",
        "throwingJournal",
        "armCare",
        "Lifting",
        "hittingJournal",
        "postPerformance",
        "nutrition"
    ]
    rows: List[Dict[str, Any]] = []
    for doc in raw_docs:
        # date to ISO
        dt_iso = doc["date"].isoformat()
        row = {"date": dt_iso}

        for sec in sections:
            v = doc.get(sec)
            if v is None:
                row[sec] = ""
            elif isinstance(v, (str, int, float, bool)):
                row[sec] = v
            else:
                # use default=str so datetimes inside dicts become strings
                row[sec] = json.dumps(v, default=str)
        rows.append(row)

    # 5) Build CSV
    header = ["date"] + sections
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)

    # 6) Stream back
    filename = f"{raw_uid}_logs_{req.start.date()}_to_{req.end.date()}.csv"
    return StreamingResponse(
        iter(buf.getvalue().splitlines(True)),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# New GET-based export for direct downloads via browser/link
@app.get("/export_csv")
def export_csv_get(userId: str, startDate: str = Query(...), endDate: str = Query(...)):
    try:
        start = datetime.fromisoformat(startDate)
        end = datetime.fromisoformat(endDate)
    except Exception:
        raise HTTPException(400, "startDate and endDate must be ISO-formatted datetimes")

    req = FilterReq(userId=userId, startDate=start, endDate=end)
    csv_text, filename = make_csv_string(req)
    buf = io.StringIO(csv_text)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "text/csv; charset=utf-8"
    }
    return StreamingResponse(iter([buf.getvalue()]), headers=headers)


def estimate_lifting_volume(lift: dict) -> float:
    # naive proxy: count number of logged exercises × 10
    # you can parse setsAndReps more accurately here
    return len(lift.get("liftingType", [])) * 10.0

@app.post("/insights")
def insights(req: FilterReq):

    # thresholds from the PDF :contentReference[oaicite:0]{index=0}
    thresholds = {
        "Visualization":    [(4, "Good"), (2, "Average"), (0, "Needs Improvement")],
        "Consistency":      [(5, "Good"), (3, "Average"), (1, "Needs Improvement"), (0, "Not Taking It Seriously")],
        "Lifting_Season":   [(3, "Good"), (2, "Average"), (0, "Needs More Focus")],
        "Lifting_Off":      [(4, "Good"), (2, "Average"), (1, "Needs More Focus")],
        "Recovery":         [(4, "Good"), (2, "Average"), (0, "Needs More Focus")],
        "Wellness":         [(6, "Excellent"), (4, "Good"), (3, "Average"), (0, "Needs More Focus")],
        # Nutrition: we'll treat score/10 similarly to Wellness
        "Nutrition":        [(8, "Excellent"), (5, "Good"), (3, "Average"), (0, "Needs Improvement")]
    }

    def categorize(value, rules):
        for cutoff, label in rules:
            if value >= cutoff:
                return label
        return rules[-1][1]

    # 1) Clean & validate userId
    raw_uid = req.userId.strip()
    if len(raw_uid) != 24:
        raise HTTPException(400, "`userId` must be a 24‑char hex string")
    try:
        user_oid = ObjectId(raw_uid)
    except Exception:
        raise HTTPException(400, "`userId` is not a valid ObjectId")

    # 2) Fetch logs directly with date filter
    query = {
        "userId": user_oid,
        "date": {
            "$gte": req.start,
            "$lte": req.end
        }
    }
    docs = list(dailylogs.find(query).sort("date", 1))

    # 3) If no docs, return defaults
    if not docs:
        return JSONResponse({
            "series": {},
            "powerRatings": {
                cat: 50 for cat in
                ["Visualization","Consistency","Lifting","Recovery","Wellness"]
            }
        })

    # 4) Build a DataFrame
    rows = []
    for log in docs:
        vis = log.get("visualization", {})
        rows.append({
            "date": log["date"],
            "visualization_time": (
                vis.get("boxBreathingTime", 0)
              + vis.get("gameEnvironmentTime", 0)
              + vis.get("gameExecutionTime", 0)
              + vis.get("pregameRoutineTime", 0)
            ),
            "log_exists": True,
            "lifting_volume": estimate_lifting_volume(log.get("Lifting", {})),
            "recovery_done": bool(log.get("armCare", {}).get("exercisesLog")),
            "readiness": log.get("dailyWellnessQuestionnaire", {})\
                                 .get("readinessToCompete", 0)
        })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 5) Build weekly series
    w = df.resample('W')
    series = {
        "Visualization": w['visualization_time']\
                            .sum()\
                            .rename("Visualization")\
                            .reset_index()\
                            .values.tolist(),
        "Consistency":   w['log_exists']\
                            .sum()\
                            .rename("Consistency")\
                            .reset_index()\
                            .values.tolist(),
        "Lifting":       w['lifting_volume']\
                            .sum()\
                            .rename("Lifting")\
                            .reset_index()\
                            .values.tolist(),
        "Recovery":      w['recovery_done']\
                            .sum()\
                            .rename("Recovery")\
                            .reset_index()\
                            .values.tolist(),
        "Wellness":      w['readiness']\
                            .mean()\
                            .rename("Wellness")\
                            .reset_index()\
                            .values.tolist(),
    }

    # 6) Compute power ratings
    days_span = max((req.end - req.start).days, 1) / 7
    targets = {
        "Visualization": 120, "Consistency": 7,
        "Lifting": 100.0,     "Recovery": 5,
        "Wellness": 9.0
    }
    scores = {
        "Visualization": min(100, df['visualization_time'].sum() / days_span / targets["Visualization"] * 100),
        "Consistency":   min(100, df['log_exists'].resample('D').max().mean() * 100),
        "Lifting":       min(100, df['lifting_volume'].sum()     / days_span / targets["Lifting"]    * 100),
        "Recovery":      min(100, df['recovery_done'].sum()      / days_span / targets["Recovery"]   * 100),
        "Wellness":      min(100, df['readiness'].mean()                                      / targets["Wellness"]  * 100),
    }
    powerRatings = {k: round(v) for k, v in scores.items()}

    # Determine current weekly averages for the *latest* week
    latest = {cat: vals[-1][1] for cat, vals in series.items() if vals}

    categorical = {}
    for cat, avg in latest.items():
        if cat == "Lifting":
            # choose season or off-season threshold
            season = "Season" if req.start.month in range(3,9) else "Off"
            key = f"Lifting_{season}"
        else:
            key = cat
        categorical[cat] = categorize(avg, thresholds[key])

        return {"series": series, "powerRatings": powerRatings}

# List all indexes to confirm connection
print("Available indexes:", pc.list_indexes().names())

# Check if the index is ready
description = pc.describe_index(index_name)
print(f"Index '{index_name}' status:", description.status)

