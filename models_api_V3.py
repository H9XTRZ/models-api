from cryptography.fernet import Fernet
from fastapi import Request
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import json
import math
import re
import uuid, sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any
import bcrypt
import jwt
import time
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
import threading
import time as t
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

security = HTTPBearer()

DB_PATH = "models_data.db"

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("❌ SECRET_KEY is missing from environment variables.")


# --- Encryption/Decryption for OpenAI Key (for browser backend) ---
raw_encryption_key = os.getenv("ENCRYPTION_KEY")
if not raw_encryption_key:
    raise RuntimeError("❌ ENCRYPTION_KEY is missing from environment variables.")

ENCRYPTION_KEY = raw_encryption_key.encode()
fernet = Fernet(ENCRYPTION_KEY)
HARDCODED_BROWSER_USE_PASSWORD = "#Oh,whenitall,itallfallsdownYeah,thistherealone,babyImtellinyouall,itallfallsdownUh,Chi-Town,standup!Oh,whenitall,itallfallsdownSouthside,SouthsideWegonsetthispartyoffrightImtellinyouall,itallfallsdownWestside,WestsideWegonsetthispartyoffrightOh,whenitallMan,Ipromise,shessoself-consciousShehasnoideawhatshedoinincollegeThatmajorthatshemajorediņdontmakenomoneyButshewontdropout,herparentslllookatherfunnyNow,tellmethataintinsecurrTheconceptofschoolseemssosecurrSophomore,threeyurrs,aintpickedacarurrShelike,Fuckit,Illjuststaydownhurranddohair.CausethatsenoughmoneytobuyherafewpairsOfnewAirs,causeherbabydaddydontreallycareShessopreciouswiththepeerpressureCouldntaffordacar,soshenamedherdaughterAlexisShehadhairsolongthatitlookedlikeweaveThenshecutitalloff,nowshelooklikeEveAndshebedealinwithsomeissuesthatyoucantbelieveSingleblackfemaleaddictedtoretail,andwellOh,whenitall,itallfallsdownAndwhenitfallsdown,whoyougoncallnow?Imtellinyouall,itallfallsdownCmon,cmon,andwhenitfallsdownOh,whenitallMan,Ipromise,Imsoself-consciousThatswhyyoualwaysseemewithatleastoneofmywatchesRolliesandPashasdonedrovemecrazyIcantevenpronouncenothin,passthatVer-say-see!ThenIspentfourhundredbucksonthisJusttobelike,Nigga,youaintuponthis.AndIcantevengotothegrocerystoreWithoutsomeOnesthatscleanandashirtwithateamItseemwelivintheAmericanDreamButthepeoplehighestupgotthelowestself-esteemTheprettiestpeopledotheugliestthingsFortheroadtorichesanddiamondringsWeshinebecausetheyhateus,flosscausetheydegradeusWetrynabuybackour40acresAndforthatpaper,lookhowlowwellstoopEvenifyouinaBenz,youstillaniggainacoupeOh"

ENCRYPTED_OPENAI_KEY = fernet.encrypt(os.getenv("OPENAI_API_KEY").encode())

raw_admin_emails = os.getenv("ADMIN_EMAILS", "")
ADMIN_EMAILS = {email.strip() for email in raw_admin_emails.split(",") if email.strip()}


def ensure_admin(user_email: str) -> None:
    if ADMIN_EMAILS and user_email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin privileges required for this action.")

# Initialize DB
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create tables if they don't exist 
# Passwords will be stored as bcrypt hashes
cursor.execute("""CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password TEXT NOT NULL
)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS tokens (
    email TEXT PRIMARY KEY,
    token TEXT NOT NULL
)""")  # Stores access tokens
cursor.execute("""CREATE TABLE IF NOT EXISTS memories (
    email TEXT PRIMARY KEY,
    memory TEXT
)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS models (
    email TEXT,
    model_name TEXT,
    init_prompt TEXT,
    messages TEXT,
    PRIMARY KEY (email, model_name)
)""")
# Create table for current model
cursor.execute("""CREATE TABLE IF NOT EXISTS current_model (
    email TEXT PRIMARY KEY,
    model_name TEXT
)""")
# Add table to store engine tunnel port per user and device
cursor.execute("""CREATE TABLE IF NOT EXISTS engine_ports (
    email TEXT,
    device_token TEXT,
    tunnel_url TEXT,
    PRIMARY KEY (email, device_token)
)""")
# Add table for model color themes
cursor.execute("""CREATE TABLE IF NOT EXISTS model_themes (
    email TEXT,
    model_name TEXT,
    background TEXT,
    e1 TEXT,
    e2 TEXT,
    e3 TEXT,
    e4 TEXT,
    voice_id TEXT,
    PRIMARY KEY (email, model_name)
)""")
# Voice catalog shared across models
cursor.execute("""CREATE TABLE IF NOT EXISTS voices (
    name TEXT PRIMARY KEY,
    voice_id TEXT NOT NULL UNIQUE,
    description TEXT,
    accent TEXT,
    language TEXT,
    gender TEXT
)""")
# Device registration table
cursor.execute("""CREATE TABLE IF NOT EXISTS devices (
    email TEXT,
    device_token TEXT,
    device_name TEXT,
    callback_url TEXT,
    port TEXT,
    PRIMARY KEY (email, device_token)
)""")
# Ensure legacy tables gain the new columns without manual migration.
try:
    cursor.execute("ALTER TABLE devices ADD COLUMN device_name TEXT")
except sqlite3.OperationalError:
    pass
try:
    cursor.execute("ALTER TABLE devices ADD COLUMN port TEXT")
except sqlite3.OperationalError:
    pass
try:
    cursor.execute("ALTER TABLE model_themes ADD COLUMN voice_id TEXT")
except sqlite3.OperationalError:
    pass
try:
    cursor.execute("ALTER TABLE voices ADD COLUMN gender TEXT")
except sqlite3.OperationalError:
    pass
cursor.execute("""CREATE TABLE IF NOT EXISTS extensions (
    email TEXT,
    command_name TEXT,
    extension_code TEXT,
    PRIMARY KEY (email, command_name)
)""")
conn.commit()

# Models
class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class MemoryUpdate(BaseModel):
    memory: str

class ModelCreate(BaseModel):
    model_name: str
    init_prompt: str
    background: str = None
    e1: str = None
    e2: str = None
    e3: str = None
    e4: str = None
    voice_id: str | None = None

class ModelChat(BaseModel):
    message: str

# --- Extension management models ---
class ExtensionUpload(BaseModel):
    command_name: str
    extension_code: str

class ExtensionCodeRequest(BaseModel):
    command_name: str

# Device registration request model
class DeviceRegistration(BaseModel):
    device_token: str
    device_name: str
    port: str
    callback_url: str | None = None


class DeviceTokenRequest(BaseModel):
    device_token: str


class DeviceRenameRequest(BaseModel):
    device_token: str
    device_name: str


class VoiceRegistration(BaseModel):
    name: str
    voice_id: str
    description: str | None = None
    accent: str | None = None
    language: str | None = None
    gender: str | None = None


class VoiceDeleteRequest(BaseModel):
    name: str | None = None
    delete_all: bool = False


def create_refresh_token(email: str):
    payload = {
        "email": email,
        "exp": int(time.time()) + 60 * 60 * 24 * 30  # 30 days
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

@app.post("/register")
def register(req: RegisterRequest):
    cursor.execute("SELECT email FROM users WHERE email = ?", (req.email,))
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="User already exists")
    hashed_pw = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (req.email, hashed_pw))
    conn.commit()
    return {"status": "User registered successfully"}

@app.post("/login")
def login(req: LoginRequest):
    cursor.execute("SELECT password FROM users WHERE email = ?", (req.email,))
    row = cursor.fetchone()
    if row and bcrypt.checkpw(req.password.encode(), row[0].encode()):
        payload = {
            "email": req.email,
            "exp": int(time.time()) + 60 * 60 * 24 * 7  # Token expires in 7 days
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        cursor.execute("REPLACE INTO tokens (email, token) VALUES (?, ?)", (req.email, token))
        cursor.execute("INSERT OR IGNORE INTO memories (email, memory) VALUES (?, '')", (req.email,))
        conn.commit()
        refresh_token = create_refresh_token(req.email)
        return {"token": token, "refresh_token": refresh_token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

class RefreshRequest(BaseModel):
    refresh_token: str

@app.post("/refresh")
def refresh_token(req: RefreshRequest):
    try:
        payload = jwt.decode(req.refresh_token, SECRET_KEY, algorithms=["HS256"])
        new_payload = {
            "email": payload["email"],
            "exp": int(time.time()) + 60 * 60 * 24 * 7  # 7 days
        }
        new_token = jwt.encode(new_payload, SECRET_KEY, algorithm="HS256")
        cursor.execute("REPLACE INTO tokens (email, token) VALUES (?, ?)", (payload["email"], new_token))
        conn.commit()
        return {"token": new_token}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload["email"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/memory")
def get_memory(user_id: str = Depends(verify_token)):
    cursor.execute("SELECT memory FROM memories WHERE email = ?", (user_id,))
    mem_row = cursor.fetchone()
    print("📀 Opened memory database:")
    return {"memory": "SYSTEM RESPONSE: You have entered the MEMORY DATABASE\nYou have entered the MEMORY DATABASE.\nHere are all the memories currently saved with the user.\nEach memory is separated by * — this marks the end of one and the start of another.\n----\n" + mem_row[0] if mem_row else ""}

@app.post("/memory/update")
def update_memory(req: MemoryUpdate, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT memory FROM memories WHERE email = ?", (user_id,))
    existing = cursor.fetchone()
    existing_memory = existing[0] if existing and existing[0] else ""
    if not existing_memory.startswith("SYSTEM RESPONSE: You have entered the MEMORY DATABASE"):
        header = "SYSTEM RESPONSE: Here are all the memories currently saved with the user.\nEach memory is separated by * — this marks the end of one and the start of another.\n----\n"
    else:
        header = ""
    updated_memory = existing_memory + req.memory.strip() + "   *\n"
    final_memory = header + updated_memory if not existing_memory else updated_memory
    cursor.execute("UPDATE memories SET memory = ? WHERE email = ?", (final_memory, user_id))
    conn.commit()
    return {"status": "Memory updated"}

@app.delete("/admin/reset-users")
def reset_users():
    cursor.execute("DELETE FROM users")
    conn.commit()
    return {"status": "All users removed (for bcrypt fix)"}

@app.post("/create_model")
def create_model(req: ModelCreate, user_id: str = Depends(verify_token)):
    cursor.execute("INSERT OR REPLACE INTO models (email, model_name, init_prompt, messages) VALUES (?, ?, ?, ?)",
                   (user_id, req.model_name, req.init_prompt, "[]"))
    cursor.execute("INSERT OR REPLACE INTO current_model (email, model_name) VALUES (?, ?)", (user_id, req.model_name))

    theme_values = (
        req.background,
        req.e1,
        req.e2,
        req.e3,
        req.e4,
        req.voice_id,
    )
    if any(theme_values):
        cursor.execute(
            """INSERT OR REPLACE INTO model_themes 
            (email, model_name, background, e1, e2, e3, e4, voice_id) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                req.model_name,
                req.background,
                req.e1,
                req.e2,
                req.e3,
                req.e4,
                req.voice_id,
            ),
        )

    conn.commit()
    return {"status": f"Model '{req.model_name}' created and activated."}

@app.post("/switch_model")
def switch_model(model_name: str, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT 1 FROM models WHERE email = ? AND model_name = ?", (user_id, model_name))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Model not found")
    cursor.execute("INSERT OR REPLACE INTO current_model (email, model_name) VALUES (?, ?)", (user_id, model_name))
    conn.commit()
    return {"status": f"Switched to model '{model_name}'"}


# ---- /chat endpoint was removed ----



# List all models for the user
@app.get("/models")
def get_models(user_id: str = Depends(verify_token)):
    cursor.execute("SELECT model_name FROM models WHERE email = ?", (user_id,))
    models = [row[0] for row in cursor.fetchall()]
    return {"models": models}

# Get the current model state
@app.get("/model_state")
def get_model_state(user_id: str = Depends(verify_token)):
    cursor.execute("SELECT model_name FROM current_model WHERE email = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No current model set.")
    model_name = row[0]

    # Fetch model init_prompt and messages
    cursor.execute(
        "SELECT init_prompt, messages FROM models WHERE email = ? AND model_name = ?",
        (user_id, model_name)
    )
    model = cursor.fetchone()
    if not model:
        raise HTTPException(status_code=404, detail="Model data missing.")
    init_prompt = model[0]
    messages = model[1]

    # Fetch theme colors for this model (if any)
    cursor.execute(
        "SELECT background, e1, e2, e3, e4, voice_id FROM model_themes WHERE email = ? AND model_name = ?",
        (user_id, model_name)
    )
    theme_row = cursor.fetchone()
    theme = None
    if theme_row and any(theme_row):
        theme = {
            "background": theme_row[0],
            "e1": theme_row[1],
            "e2": theme_row[2],
            "e3": theme_row[3],
            "e4": theme_row[4],
        }
        voice_id = theme_row[5]
        if voice_id:
            theme["voice"] = voice_id

    return {
        "model_name": model_name,
        "init_prompt": init_prompt,
        "messages": messages,
        "theme": theme,  # None if no theme saved yet
    }
# Delete a model
# Delete a model
@app.delete("/delete_model")
def delete_model(model_name: str, user_id: str = Depends(verify_token)):
    cursor.execute("DELETE FROM models WHERE email = ? AND model_name = ?", (user_id, model_name))
    cursor.execute("DELETE FROM current_model WHERE email = ? AND model_name = ?", (user_id, model_name))
    conn.commit()
    return {"status": f"Model '{model_name}' deleted."}


@app.get("/voices")
def list_voices(user_id: str = Depends(verify_token)):
    cursor.execute("SELECT name, voice_id, description, accent, language, gender FROM voices ORDER BY name")
    voices = [
        {
            "name": row[0],
            "voice_id": row[1],
            "description": row[2],
            "accent": row[3],
            "language": row[4],
            "gender": row[5],
        }
        for row in cursor.fetchall()
    ]
    return {"voices": voices}


@app.post("/admin/voices")
def register_voice(req: VoiceRegistration, user_id: str = Depends(verify_token)):
    ensure_admin(user_id)
    cursor.execute(
        "INSERT OR REPLACE INTO voices (name, voice_id, description, accent, language, gender) VALUES (?, ?, ?, ?, ?, ?)",
        (req.name, req.voice_id, req.description, req.accent, req.language, req.gender),
    )
    conn.commit()
    return {"status": f"Voice '{req.name}' registered."}


@app.delete("/admin/voices")
def delete_voice(req: VoiceDeleteRequest, user_id: str = Depends(verify_token)):
    ensure_admin(user_id)
    if req.delete_all:
        cursor.execute("DELETE FROM voices")
        conn.commit()
        return {"status": "All voices removed."}
    if not req.name:
        raise HTTPException(status_code=400, detail="Provide a voice name to remove or set delete_all to true.")
    cursor.execute("DELETE FROM voices WHERE name = ?", (req.name,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Voice not found.")
    conn.commit()
    return {"status": f"Voice '{req.name}' removed."}

# Get theme colors for a specific model
@app.get("/get_theme")
def get_theme(model_name: str, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT background, e1, e2, e3, e4 FROM model_themes WHERE email = ? AND model_name = ?", (user_id, model_name))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Theme not found for this model")
    return {
        "background": row[0],
        "e1": row[1],
        "e2": row[2],
        "e3": row[3],
        "e4": row[4]
    }


# 🔐 Returns encrypted OpenAI key for use by browser backend after verifying password
class EncryptedKeyRequest(BaseModel):
    password: str

@app.post("/get_encrypted_openai_key")
def get_encrypted_openai_key(req: EncryptedKeyRequest):
    if req.password != HARDCODED_BROWSER_USE_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid password")
    return {"encrypted_key": ENCRYPTED_OPENAI_KEY.decode()}



    
      



# Device registration endpoint
@app.post("/register_device")
def register_device(req: DeviceRegistration, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT 1 FROM devices WHERE email = ? AND device_token = ?", (user_id, req.device_token))
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="Device token already registered for this user.")
    cursor.execute(
        "INSERT INTO devices (email, device_token, device_name, callback_url, port) VALUES (?, ?, ?, ?, ?)",
        (user_id, req.device_token, req.device_name, req.callback_url, req.port)
    )
    conn.commit()
    return {"status": "Device registered successfully"}


@app.post("/unregister_device")
def unregister_device(req: DeviceTokenRequest, user_id: str = Depends(verify_token)):
    cursor.execute("DELETE FROM devices WHERE email = ? AND device_token = ?", (user_id, req.device_token))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Device not found for this user.")
    conn.commit()
    return {"status": "Device unregistered successfully"}


@app.post("/rename_device")
def rename_device(req: DeviceRenameRequest, user_id: str = Depends(verify_token)):
    cursor.execute(
        "UPDATE devices SET device_name = ? WHERE email = ? AND device_token = ?",
        (req.device_name, user_id, req.device_token)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Device not found for this user.")
    conn.commit()
    return {"status": "Device renamed successfully"}


@app.get("/get_registered_devices")
def get_registered_devices(user_id: str = Depends(verify_token)):
    cursor.execute("SELECT device_token, device_name, port FROM devices WHERE email = ?", (user_id,))
    rows = cursor.fetchall()
    devices = {token: (name if name else token) for token, name, _ in rows}
    device_ports = {token: port for token, _, port in rows if port}
    response = {"devices": devices}
    if device_ports:
        response["device_ports"] = device_ports
    return response



# ---- Engine Port Endpoints ----

class EnginePortUpdate(BaseModel):
    tunnel_url: str
    device_token: str

@app.post("/update_engine_port")
def update_engine_port(req: EnginePortUpdate, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT 1 FROM devices WHERE email = ? AND device_token = ?", (user_id, req.device_token))
    if not cursor.fetchone():
        raise HTTPException(status_code=403, detail="Device not registered.")
    cursor.execute("REPLACE INTO engine_ports (email, device_token, tunnel_url) VALUES (?, ?, ?)",
                   (user_id, req.device_token, req.tunnel_url))
    conn.commit()
    return {"status": "Tunnel port updated"}

@app.get("/get_engine_port")
def get_engine_port(device_token: str, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT tunnel_url FROM engine_ports WHERE email = ? AND device_token = ?", (user_id, device_token))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No tunnel port found for this user/device.")
    return {"tunnel_url": row[0]}
# ---- Message Sync Endpoints ----


def _coerce_message_list(payload: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [msg for msg in payload if isinstance(msg, dict)]
    if isinstance(payload, str):
        if not payload.strip():
            return []
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Messages payload must be a JSON array.") from exc
        if isinstance(data, list):
            return [msg for msg in data if isinstance(msg, dict)]
    raise HTTPException(status_code=400, detail="Messages payload must be a list or JSON string.")


def _deserialize_stored_messages(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, "", "null"):
        return []
    if isinstance(raw, list):
        source = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        source = parsed if isinstance(parsed, list) else []
    else:
        return []
    return [msg for msg in source if isinstance(msg, dict)]


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def _normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    normalized: list[dict[str, Any]] = []
    changed = False
    now = datetime.now(timezone.utc)
    fallback_index = 0

    for message in messages:
        if not isinstance(message, dict):
            changed = True
            continue
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            changed = True
            continue
        timestamp = _parse_timestamp(message.get("timestamp"))
        if timestamp is None:
            timestamp = now + timedelta(milliseconds=fallback_index)
            fallback_index += 1
            changed = True
        canonical_ts = timestamp.astimezone(timezone.utc).isoformat()
        if message.get("timestamp") != canonical_ts:
            changed = True
        entry = dict(message)
        entry["timestamp"] = canonical_ts
        normalized.append(entry)
    return normalized, changed


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, ensure_ascii=False)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _vectorize(text: str) -> Counter:
    return Counter(_tokenize(text))


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    if not vec_a or not vec_b:
        return 0.0
    intersection = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[token] * vec_b[token] for token in intersection)
    magnitude_a = math.sqrt(sum(count * count for count in vec_a.values()))
    magnitude_b = math.sqrt(sum(count * count for count in vec_b.values()))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def _context_cutoff(length: str) -> datetime:
    if not length:
        raise HTTPException(status_code=400, detail="context_length is required.")
    value = length.strip().lower()
    if value == "all":
        return datetime.min.replace(tzinfo=timezone.utc)
    match = re.fullmatch(r"(\d+)\s*([smhdwmy])", value)
    if not match:
        raise HTTPException(status_code=400, detail="context_length must follow format like '1m' or '2w'.")
    amount = int(match.group(1))
    unit = match.group(2)
    if amount < 0:
        raise HTTPException(status_code=400, detail="context_length value must be positive.")
    if unit == "s":
        delta = timedelta(seconds=amount)
    elif unit == "m":
        delta = timedelta(days=30 * amount)
    elif unit == "h":
        delta = timedelta(hours=amount)
    elif unit == "d":
        delta = timedelta(days=amount)
    elif unit == "w":
        delta = timedelta(weeks=amount)
    elif unit == "y":
        delta = timedelta(days=365 * amount)
    else:
        raise HTTPException(status_code=400, detail="Unsupported context_length unit.")
    return datetime.now(timezone.utc) - delta


def _score_messages(query: str, candidates: list[tuple[dict[str, Any], datetime]]) -> list[tuple[float, datetime, dict[str, Any]]]:
    query_vector = _vectorize(query)
    scored: list[tuple[float, datetime, dict[str, Any]]] = []
    for message, timestamp in candidates:
        text = f"{message.get('role', '')} {message.get('content', '')}"
        score = _cosine_similarity(query_vector, _vectorize(text))
        scored.append((score, timestamp, message))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:4]
    top.sort(key=lambda item: item[1])
    return top


# Request model for syncing messages
class MessageSyncRequest(BaseModel):
    messages: list[dict[str, Any]] | str  # Accept raw list or JSON string

# POST endpoint to sync message history for the current model
@app.post("/sync_messages")
def sync_messages(req: MessageSyncRequest, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT model_name FROM current_model WHERE email = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No current model set.")
    model_name = row[0]
    incoming = _coerce_message_list(req.messages)
    normalized, _ = _normalize_messages(incoming)
    serialized = _serialize_messages(normalized)
    cursor.execute(
        "UPDATE models SET messages = ? WHERE email = ? AND model_name = ?",
        (serialized, user_id, model_name),
    )
    conn.commit()
    return {"status": "Messages synced", "messages": serialized}

# GET endpoint to retrieve the current model's messages
@app.get("/sync_messages")
def get_synced_messages(user_id: str = Depends(verify_token)):
    cursor.execute("SELECT model_name FROM current_model WHERE email = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No current model set.")
    model_name = row[0]
    cursor.execute("SELECT messages FROM models WHERE email = ? AND model_name = ?", (user_id, model_name))
    model = cursor.fetchone()
    if not model:
        raise HTTPException(status_code=404, detail="Model data missing.")
    return {"messages": model[0]}


class ContextRequest(BaseModel):
    context_length: str
    model_name: str = Field(alias="Model")
    user_message: str

    class Config:
        allow_population_by_field_name = True


@app.post("/get_context")
def get_context(req: ContextRequest, user_id: str = Depends(verify_token)):
    model_name = req.model_name
    cursor.execute(
        "SELECT messages FROM models WHERE email = ? AND model_name = ?",
        (user_id, model_name),
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found.")

    stored_messages = _deserialize_stored_messages(row[0])
    normalized_messages, changed = _normalize_messages(stored_messages)
    if changed:
        serialized = _serialize_messages(normalized_messages)
        cursor.execute(
            "UPDATE models SET messages = ? WHERE email = ? AND model_name = ?",
            (serialized, user_id, model_name),
        )
        conn.commit()

    cutoff = _context_cutoff(req.context_length)
    candidates: list[tuple[dict[str, Any], datetime]] = []
    for message in normalized_messages:
        timestamp = _parse_timestamp(message.get("timestamp"))
        if not timestamp:
            continue
        if timestamp >= cutoff:
            candidates.append((message, timestamp.astimezone(timezone.utc)))

    if not candidates:
        return {
            "model": model_name,
            "context_length": req.context_length,
            "messages": [],
        }

    ranked = _score_messages(req.user_message, candidates)
    relevant_messages = []
    for score, timestamp, message in ranked:
        enriched = dict(message)
        enriched["similarity"] = round(score, 4)
        enriched["timestamp"] = timestamp.isoformat()
        relevant_messages.append(enriched)

    return {
        "model": model_name,
        "context_length": req.context_length,
        "messages": relevant_messages,
    }


# --- Extension Management Endpoints ---

@app.post("/upload_extension")
def upload_extension(req: ExtensionUpload, user_id: str = Depends(verify_token)):
    cursor.execute("REPLACE INTO extensions (email, command_name, extension_code) VALUES (?, ?, ?)",
                   (user_id, req.command_name, req.extension_code))
    conn.commit()
    return {"status": f"Extension '{req.command_name}' uploaded successfully."}

@app.get("/get_user_extensions")
def get_user_extensions(user_id: str = Depends(verify_token)):
    cursor.execute("SELECT command_name FROM extensions WHERE email = ?", (user_id,))
    extensions = [row[0] for row in cursor.fetchall()]
    return {"extensions": extensions}

@app.post("/get_extension_code")
def get_extension_code(req: ExtensionCodeRequest, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT extension_code FROM extensions WHERE email = ? AND command_name = ?",
                   (user_id, req.command_name))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Extension not found")
    return {"extension_code": row[0]}
#again
