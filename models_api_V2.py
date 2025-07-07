from cryptography.fernet import Fernet
from fastapi import Request
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uuid, sqlite3
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
    PRIMARY KEY (email, model_name)
)""")
# Device registration table
cursor.execute("""CREATE TABLE IF NOT EXISTS devices (
    email TEXT,
    device_token TEXT,
    callback_url TEXT,
    PRIMARY KEY (email, device_token)
)""")
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
    callback_url: str


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
        header = "SYSTEM RESPONSE: You have entered the MEMORY DATABASE\nYou have entered the MEMORY DATABASE.\nHere are all the memories currently saved with the user.\nEach memory is separated by * — this marks the end of one and the start of another.\n----\n"
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

    # Insert theme colors if all are provided
    if all([req.background, req.e1, req.e2, req.e3, req.e4]):
        cursor.execute("""INSERT OR REPLACE INTO model_themes 
            (email, model_name, background, e1, e2, e3, e4) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, req.model_name, req.background, req.e1, req.e2, req.e3, req.e4))

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
    cursor.execute("SELECT init_prompt, messages FROM models WHERE email = ? AND model_name = ?", (user_id, model_name))
    model = cursor.fetchone()
    if not model:
        raise HTTPException(status_code=404, detail="Model data missing.")
    return {"model_name": model_name , "init_prompt": model[0]}   # add this to it to get the init prompt and chat history for the current model: , "init_prompt": model[0], "messages": model[1]

# Delete a model
# Delete a model
@app.delete("/delete_model")
def delete_model(model_name: str, user_id: str = Depends(verify_token)):
    cursor.execute("DELETE FROM models WHERE email = ? AND model_name = ?", (user_id, model_name))
    cursor.execute("DELETE FROM current_model WHERE email = ? AND model_name = ?", (user_id, model_name))
    conn.commit()
    return {"status": f"Model '{model_name}' deleted."}

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
    cursor.execute("REPLACE INTO devices (email, device_token, callback_url) VALUES (?, ?, ?)", (user_id, req.device_token, req.callback_url))
    conn.commit()
    return {"status": "Device registered successfully"}



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

# Request model for syncing messages
class MessageSyncRequest(BaseModel):
    messages: str  # Should be a JSON stringified list of message dicts

# POST endpoint to sync message history for the current model
@app.post("/sync_messages")
def sync_messages(req: MessageSyncRequest, user_id: str = Depends(verify_token)):
    cursor.execute("SELECT model_name FROM current_model WHERE email = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No current model set.")
    model_name = row[0]
    cursor.execute("UPDATE models SET messages = ? WHERE email = ? AND model_name = ?", (req.messages, user_id, model_name))
    conn.commit()
    return {"status": "Messages synced", "messages": req.messages}

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