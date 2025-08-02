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

# ─── Chat Endpoint ──────────────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):

    text = req.message.lower()

    # 1) Download Intent
    if "download" in text and "data" in text:
        days = 7
        if "30" in text or "thirty" in text:
            days = 30
        now = datetime.utcnow()
        start = now - timedelta(days=days)
        end   = now

        csv_req = FilterReq(userId=req.userId, startDate=start, endDate=end)

        # generate the CSV text & filename
        csv_text, filename = make_csv_string(csv_req)

        # optionally base64-encode if you want safe JSON transport
        csv_b64 = base64.b64encode(csv_text.encode()).decode()

        return {
            "tag":      "csv_download",
            "reply":    csv_text
        }


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
     "You are a data‑driven baseball coach AI. Use ONLY the provided data context to "
     "answer questions about training, performance, recovery, wellness, and nutrition. "
     "Refer to prior conversation for follow‑ups."},
    {"role":"system","content": context_block}
]

    # Assemble and call
    messages = system_msgs + msg_history + [{"role":"user","content":req.message}]
    comp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=500
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
def export_csv_get(
    userId:    str,
    startDate: str = Query(..., description="ISO format, e.g. 2025-05-01T00:00:00"),
    endDate:   str = Query(..., description="ISO format, e.g. 2025-05-31T23:59:59"),
):
    # 1) Parse dates
    try:
        start = datetime.fromisoformat(startDate)
        end   = datetime.fromisoformat(endDate)
    except ValueError:
        raise HTTPException(400, "startDate and endDate must be ISO‑formatted datetimes")

    # 2) Build a FilterReq to reuse the POST logic
    req = FilterReq(userId=userId, startDate=start, endDate=end)

    # 3) Delegate to the POST handler
    return export_csv(req)

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

