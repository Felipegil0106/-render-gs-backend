"""
Backend Render-GS (2DGS) — Orquestación de Pods GPU para PRUEBAS
═══════════════════════════════════════════════════════════════════
Backend SEPARADO del viejo. Misma lógica probada (cascada de GPUs,
watchdog, HMAC, terminación de pod), pero apunta al motor NUEVO 2DGS:
  - Imagen:  felipegil0106/render-gs:v2  (2DGS + COLMAP)
  - Worker:  render-gs-worker/worker_gs.py
  - Entrega: malla CRUDA (.ply) para medir la geometría (paredes/cuadrado)

Flujo para el usuario: entra a la URL → sube el ZIP → se alquila sola la
RTX 4090 (o la siguiente en la jerarquía) → renderiza → descarga el .ply.

NO toca el backend viejo. Es para probar cada avance del nuevo motor.

Endpoints:
  GET  /                              → página (subir ZIP)
  POST /api/jobs                      → recibe ZIP, alquila Pod, devuelve job_id
  GET  /api/jobs/{id}                 → estado del job
  GET  /api/jobs/{id}/download        → URL del .ply terminado
  GET  /api/jobs/{id}/log             → log del worker (texto plano)
  POST /api/internal/callback/{id}    → callback HMAC del pod
  GET  /api/health                    → health check
"""
import os, uuid, json, hmac, hashlib, sqlite3, asyncio
from datetime import datetime, timezone
from contextlib import contextmanager
import boto3
from botocore.config import Config
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
RUNPOD_API_KEY      = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_API_KEY_2    = os.environ.get("RUNPOD_API_KEY_2", "")
RUNPOD_API_URL      = "https://api.runpod.io/graphql"
# API REST nueva de RunPod (la recomendada). Resuelve el "Sin GPU disponible"
# probando varias GPUs y datacenters automáticamente, como hace la web.
RUNPOD_REST_URL     = "https://rest.runpod.io/v1"
# Lista de GPUs EN ORDEN JERÁRQUICO (igual que tu backend viejo: de mejor a
# aceptable). Con gpuTypePriority='availability', RunPod recorre esta lista
# DE ARRIBA HACIA ABAJO y alquila la PRIMERA que tenga stock. Es decir, SIEMPRE
# intenta la 4090 primero; si no hay, la A6000; si no, la 6000 Ada; y así.
# El orden de esta lista ES la jerarquía y se respeta siempre.
GPU_FALLBACK_LIST = [
    "NVIDIA GeForce RTX 4090",            # 1º preferida
    "NVIDIA RTX A6000",                   # 2º
    "NVIDIA RTX 6000 Ada Generation",     # 3º
    "NVIDIA L40S",                        # 4º
    "NVIDIA L40",                         # 5º
    "NVIDIA A100 80GB PCIe",              # 6º
    "NVIDIA A100-SXM4-80GB",              # 7º
    "NVIDIA A100-SXM4-40GB",              # 8º
    "NVIDIA A40",                         # 9º
    "NVIDIA GeForce RTX 3090 Ti",         # 10º
    "NVIDIA GeForce RTX 3090",            # 11º
    "NVIDIA RTX A5000",                   # 12º
    "NVIDIA L4",                          # 13º
]
RUNPOD_CUENTA_1_NOMBRE = os.environ.get("RUNPOD_CUENTA_1_NOMBRE", "Cuenta 1")
RUNPOD_CUENTA_2_NOMBRE = os.environ.get("RUNPOD_CUENTA_2_NOMBRE", "Cuenta 2")

def _runpod_cuentas_disponibles():
    cuentas = []
    if RUNPOD_API_KEY:
        cuentas.append({"id": "1", "nombre": RUNPOD_CUENTA_1_NOMBRE})
    if RUNPOD_API_KEY_2:
        cuentas.append({"id": "2", "nombre": RUNPOD_CUENTA_2_NOMBRE})
    return cuentas

def _runpod_key_de_cuenta(cuenta_id):
    if str(cuenta_id) == "2" and RUNPOD_API_KEY_2:
        return RUNPOD_API_KEY_2
    return RUNPOD_API_KEY

# ── CAMBIO 1: imagen NUEVA (2DGS + COLMAP) en vez de la vieja ──
RUNPOD_IMAGE        = os.environ.get("RUNPOD_IMAGE", "felipegil0106/render-gs:v2")

# Config del pod — AJUSTADO para caber en 4090 con 40GB de disco total.
# Antes pedíamos 60+100=160GB y RunPod rechazaba las 4090 (solo tienen 40GB).
# Para una habitación con 2DGS, 30GB de container sobra; volumen 0 (todo el
# trabajo va en el container, no necesitamos disco persistente).
# Config del pod — IGUAL que el backend viejo que SÍ conseguía la 4090.
# (El viejo usaba 50 container + 100 volumen y funcionaba; lo replicamos.)
POD_CONTAINER_DISK_GB = 50
POD_VOLUME_DISK_GB    = 100
POD_MIN_VCPU          = 8
POD_MIN_MEMORY_GB     = 32

# ── Jerarquía de GPUs (IDÉNTICA a tu backend viejo) ──
GPU_PREFERENCES = [
    ("RTX 4090",        ["4090"],                ["pro", "ti", "mobile"]),
    ("RTX A6000",       ["a6000"],               ["ada"]),
    ("RTX 6000 Ada",    ["6000", "ada"],         ["pro"]),
    ("L40S",            ["l40s"],                []),
    ("L40",             ["l40"],                 ["l40s"]),
    ("A100 SXM 80GB",   ["a100", "sxm", "80"],   []),
    ("A100 PCIe 80GB",  ["a100", "pcie", "80"],  ["sxm"]),
    ("A100 80GB",       ["a100", "80"],          ["sxm", "pcie"]),
    ("A100 PCIe 40GB",  ["a100", "pcie", "40"],  ["80", "sxm"]),
    ("A100 40GB",       ["a100", "40"],          ["80", "sxm", "pcie"]),
    ("A40",             ["a40"],                 ["rtx", "ada"]),
    ("RTX 3090 Ti",     ["3090", "ti"],          []),
    ("RTX 3090",        ["3090"],                ["ti"]),
    ("RTX A5000",       ["a5000"],               ["ada"]),
    ("L4",              ["l4"],                  ["l40", "l4d"]),
    ("RTX A4500",       ["a4500"],               ["pro", "ada"]),
    ("RTX A4000",       ["a4000"],               ["pro", "ada"]),
]
# Disco por GPU (igual que el backend viejo que funcionaba).
GPU_DISK_CONFIG = {
    "A100 SXM 80GB":   {"container": 60, "volume": 120},
    "A100 PCIe 80GB":  {"container": 60, "volume": 120},
    "A100 80GB":       {"container": 60, "volume": 120},
    "A100 PCIe 40GB":  {"container": 60, "volume": 120},
    "A100 40GB":       {"container": 60, "volume": 120},
    "RTX A4500":       {"container": 60, "volume": 80},
    "RTX A4000":       {"container": 60, "volume": 80},
}
BANNED_GPU_KEYWORDS = ["5090", "rtx pro", "b200", "b300", "h100", "h200"]

# ── CAMBIO 2: worker NUEVO (2DGS) ──
WORKER_SCRIPT_URL   = os.environ.get(
    "WORKER_SCRIPT_URL",
    "https://raw.githubusercontent.com/Felipegil0106/render-gs-worker/main/worker_gs.py")

# R2
R2_ACCOUNT_ID  = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY  = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY  = os.environ.get("R2_SECRET_KEY", "")
R2_BUCKET      = os.environ.get("R2_BUCKET", "gaussian-scanner")

BACKEND_URL    = os.environ.get("BACKEND_URL", "")
CALLBACK_SECRET = os.environ.get("CALLBACK_SECRET", "")
if not CALLBACK_SECRET:
    CALLBACK_SECRET = uuid.uuid4().hex
    print("[WARN] CALLBACK_SECRET no estaba seteado, generado uno nuevo")

# Watchdog (igual que el viejo)
POD_MAX_LIFETIME_MIN      = 90
POD_HEARTBEAT_TIMEOUT_MIN = 30
WATCHDOG_INTERVAL_SEC     = 300
PORT = int(os.environ.get("PORT", "8000"))
DB_PATH = os.environ.get("DB_PATH", "/data/jobs_gs.db")

# ══════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="Render-GS API (2DGS)", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════
# BASE DE DATOS
# ══════════════════════════════════════════════════════════════
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'uploading',
                quality TEXT DEFAULT 'fast',
                pod_id TEXT,
                gpu_type TEXT,
                created_at TEXT,
                updated_at TEXT,
                last_heartbeat TEXT,
                progress REAL DEFAULT 0,
                message TEXT,
                frames_used INTEGER,
                ply_mb REAL,
                error TEXT,
                seconds REAL,
                ply_key TEXT,
                worker_log TEXT,
                runpod_cuenta TEXT
            )
        """)
        for col, typ in [("pod_id","TEXT"),("gpu_type","TEXT"),
                         ("last_heartbeat","TEXT"),("progress","REAL"),
                         ("message","TEXT"),("worker_log","TEXT"),
                         ("ply_mb","REAL"),("runpod_cuenta","TEXT")]:
            try: db.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError: pass

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def job_update(job_id, **fields):
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    cols = ", ".join(f"{k}=?" for k in fields.keys())
    vals = list(fields.values()) + [job_id]
    with get_db() as db:
        db.execute(f"UPDATE jobs SET {cols} WHERE id=?", vals)

def job_get(job_id):
    with get_db() as db:
        r = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None

# ══════════════════════════════════════════════════════════════
# R2 CLIENTE
# ══════════════════════════════════════════════════════════════
def get_r2():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def r2_put_url(key, expires=7200):
    return get_r2().generate_presigned_url(
        "put_object", Params={"Bucket":R2_BUCKET,"Key":key}, ExpiresIn=expires)

def r2_get_url(key, expires=7200):
    return get_r2().generate_presigned_url(
        "get_object", Params={"Bucket":R2_BUCKET,"Key":key}, ExpiresIn=expires)

def r2_upload_file(file_obj, key):
    get_r2().upload_fileobj(file_obj, R2_BUCKET, key)

# ══════════════════════════════════════════════════════════════
# BOOTSTRAP del pod (descarga worker_gs.py y lo corre)
# ══════════════════════════════════════════════════════════════
def _bootstrap_body() -> str:
    """El comando de arranque SIN el envoltorio 'bash -lc'. Lo usan tanto el
    bootstrap de GraphQL (envuelto) como la REST API (en dockerStartCmd)."""
    return (
        "set -e; "
        "echo \"[bootstrap] iniciando (imagen render-gs:v2 con 2DGS+COLMAP)\"; "
        "which colmap && echo \"[bootstrap] colmap OK\" || echo \"[bootstrap] WARN: falta colmap\"; "
        "python -c \"import diff_surfel_rasterization; print(\\\"[bootstrap] 2DGS OK\\\")\" || echo \"[bootstrap] WARN: falta 2DGS\"; "
        "pip install --no-cache-dir boto3==1.34.34 >/dev/null 2>&1 || pip install boto3; "
        "mkdir -p /workspace; "
        f"wget -q -O /workspace/worker_gs.py \"{WORKER_SCRIPT_URL}\"; "
        "echo \"[bootstrap] worker descargado, ejecutando\"; "
        "cd /workspace && python -u worker_gs.py"
    )

def build_bootstrap() -> str:
    """Comando de arranque del pod para GraphQL (envuelto en bash -lc)."""
    return "bash -lc '" + _bootstrap_body() + "'"

# ══════════════════════════════════════════════════════════════
# RUNPOD GRAPHQL (IDÉNTICO a tu backend viejo)
# ══════════════════════════════════════════════════════════════
class RunPod:
    _api_key_activa = RUNPOD_API_KEY

    @staticmethod
    def set_cuenta(cuenta_id):
        RunPod._api_key_activa = _runpod_key_de_cuenta(cuenta_id)

    @staticmethod
    async def _query(query, variables=None):
        key = RunPod._api_key_activa or RUNPOD_API_KEY
        headers = {"Content-Type":"application/json",
                   "Authorization":f"Bearer {key}"}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(RUNPOD_API_URL, headers=headers,
                            json={"query":query, "variables":variables or {}})
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"RunPod GraphQL: {data['errors']}")
            return data["data"]

    @staticmethod
    def _matches(display_name, keywords_req, keywords_forbidden):
        n = display_name.lower()
        if any(b in n for b in BANNED_GPU_KEYWORDS):
            return False
        if not all(k in n for k in keywords_req):
            return False
        if any(f in n for f in keywords_forbidden):
            return False
        return True

    @staticmethod
    async def list_all_gpus():
        try:
            data = await RunPod._query("""
                query { gpuTypes { id displayName memoryInGb } }
            """)
            return data.get("gpuTypes", []) or []
        except Exception as e:
            print(f"[runpod] list_all_gpus error: {e}")
            return []

    @staticmethod
    def disk_config_for(label):
        return GPU_DISK_CONFIG.get(label, {
            "container": POD_CONTAINER_DISK_GB,
            "volume": POD_VOLUME_DISK_GB,
        })

    @staticmethod
    async def create_pod_rest(job_id, env_vars):
        """SOLUCIÓN DEFINITIVA al 'Sin GPU disponible': crea el pod con la API
        REST nueva de RunPod, mandando TODA la lista de GPUs juntas y dejando
        que RunPod elija la primera disponible en cualquier datacenter (igual
        que la web). Así no falla por pedir una sola GPU/config rígida.
        Devuelve (pod_dict, gpu_usada). Lanza error solo si NINGUNA GPU de la
        lista tiene stock en ninguna nube."""
        key = RunPod._api_key_activa or RUNPOD_API_KEY
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        bootstrap = build_bootstrap()
        # env como OBJETO (en REST, no lista).
        env_obj = {k: str(v) for k, v in env_vars.items()}
        # Probamos primero SECURE, y si no hay en ninguna GPU, COMMUNITY.
        ultimos_errores = []
        for cloud in ("SECURE", "COMMUNITY"):
            body = {
                "name": f"render-gs-{job_id[:8]}",
                "imageName": RUNPOD_IMAGE,
                "cloudType": cloud,
                "computeType": "GPU",
                "gpuTypeIds": GPU_FALLBACK_LIST,        # lista → fallback auto
                "gpuTypePriority": "availability",      # la 1ª disponible
                "gpuCount": 1,
                "containerDiskInGb": POD_CONTAINER_DISK_GB,
                "volumeInGb": POD_VOLUME_DISK_GB,
                "volumeMountPath": "/workspace",
                "minRAMPerGPU": 20,                     # por GPU (REST), no total
                "minVCPUPerGPU": 4,                     # por GPU (REST), no total
                "dataCenterPriority": "availability",   # cualquier datacenter
                "ports": ["8888/http"],
                "env": env_obj,
                # dockerArgs (GraphQL) → en REST el arranque va en dockerStartCmd.
                "dockerStartCmd": ["bash", "-lc", _bootstrap_body()],
            }
            try:
                async with httpx.AsyncClient(timeout=60) as c:
                    r = await c.post(f"{RUNPOD_REST_URL}/pods", headers=headers, json=body)
                if r.status_code in (200, 201):
                    data = r.json()
                    gpu_usada = "?"
                    # La respuesta REST trae la GPU asignada en machine/gpu.
                    try:
                        gpu_usada = (data.get("machine", {}) or {}).get("gpuTypeId") \
                                    or (data.get("gpu", {}) or {}).get("id") \
                                    or "GPU asignada"
                    except Exception:
                        pass
                    print(f"[rest] pod creado en {cloud}: {data.get('id')} ({gpu_usada})")
                    return data, gpu_usada
                else:
                    msg = f"{cloud} HTTP {r.status_code}: {r.text[:300]}"
                    print(f"[rest] {msg}")
                    ultimos_errores.append(msg)
            except Exception as e:
                msg = f"{cloud} excepción: {e}"
                print(f"[rest] {msg}")
                ultimos_errores.append(msg)
        raise RuntimeError(
            "Ninguna GPU de la lista tiene stock (probado SECURE y COMMUNITY). "
            f"Detalle: {' | '.join(ultimos_errores[:4])}"
        )

    @staticmethod
    async def create_pod(job_id, gpu_type_id, env_vars, container_gb=None, volume_gb=None):
        container_gb = container_gb or POD_CONTAINER_DISK_GB
        volume_gb = volume_gb or POD_VOLUME_DISK_GB
        bootstrap = build_bootstrap()
        env_list = [{"key":k, "value":str(v)} for k, v in env_vars.items()]
        mutation = """
        mutation deploy($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id machineId desiredStatus
            }
        }
        """
        variables = {"input": {
            "cloudType":"SECURE",
            "gpuCount":1,
            "volumeInGb": volume_gb,
            "containerDiskInGb": container_gb,
            "minVcpuCount": POD_MIN_VCPU,
            "minMemoryInGb": POD_MIN_MEMORY_GB,
            "gpuTypeId": gpu_type_id,
            "name": f"render-gs-{job_id[:8]}",
            "imageName": RUNPOD_IMAGE,
            "dockerArgs": bootstrap,
            "ports":"8888/http",
            "volumeMountPath":"/workspace",
            "env": env_list,
        }}
        print(f"[runpod] Pod create: gpu={gpu_type_id} cont={container_gb}GB vol={volume_gb}GB")
        data = await RunPod._query(mutation, variables)
        return data["podFindAndDeployOnDemand"]

    @staticmethod
    async def terminate_pod(pod_id):
        """Destruye el pod (REST API DELETE). Deja de cobrar."""
        if not pod_id: return False
        key = RunPod._api_key_activa or RUNPOD_API_KEY
        headers = {"Authorization": f"Bearer {key}"}
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.delete(f"{RUNPOD_REST_URL}/pods/{pod_id}", headers=headers)
            print(f"[rest] Pod {pod_id} terminado (HTTP {r.status_code})")
            return True
        except Exception as e:
            print(f"[rest] terminate falló: {e}")
            # Respaldo: intentar por GraphQL (por si REST falla).
            try:
                await RunPod._query("""
                    mutation t($input: PodTerminateInput!) { podTerminate(input: $input) }
                """, {"input":{"podId":pod_id}})
                print(f"[runpod] Pod {pod_id} terminado (GraphQL respaldo)")
                return True
            except Exception as e2:
                print(f"[runpod] terminate GraphQL también falló: {e2}")
                return False

    @staticmethod
    async def list_my_pods():
        try:
            data = await RunPod._query("""
                query { myself { pods { id name desiredStatus runtime { uptimeInSeconds } } } }
            """)
            return data.get("myself", {}).get("pods", []) or []
        except Exception:
            return []

# ══════════════════════════════════════════════════════════════
# PROVISIÓN DE GPU — cascada IDÉNTICA (4090 → segunda mejor)
# ══════════════════════════════════════════════════════════════
def _tag_pod(provider, pid):
    return f"{provider}:{pid}"

def _split_pod_tag(tagged):
    if not tagged:
        return ("runpod", "")
    if ":" in tagged:
        prov, _, pid = tagged.partition(":")
        if prov in ("runpod",):
            return (prov, pid)
    return ("runpod", tagged)

async def provision_gpu(job_id, env_vars, cuenta_id="1"):
    """SOLUCIÓN DEFINITIVA: usa la API REST de RunPod, que prueba TODA la lista
    de GPUs (4090 → A5000 → 3090 → A6000 → …) en SECURE y luego COMMUNITY,
    en cualquier datacenter, y alquila la primera disponible. Esto replica lo
    que hace la web, así que ya no falla por pedir una sola GPU rígida.
    Devuelve (pod_tagged_id, provider, gpu_label, gpu_display, disk)."""
    RunPod.set_cuenta(cuenta_id)
    key_activa = _runpod_key_de_cuenta(cuenta_id)
    print(f"[provision] usando RunPod cuenta {cuenta_id} (REST API con fallback)")
    if not key_activa:
        raise RuntimeError(f"La cuenta {cuenta_id} no tiene API key configurada.")
    pod, gpu_usada = await RunPod.create_pod_rest(job_id, env_vars)
    pid = pod.get("id", "")
    disk = {"container": POD_CONTAINER_DISK_GB, "volume": POD_VOLUME_DISK_GB}
    return _tag_pod("runpod", pid), "runpod", str(gpu_usada), str(gpu_usada), disk

async def terminate_any(pod_tagged_id, cuenta_id="1"):
    provider, pid = _split_pod_tag(pod_tagged_id)
    RunPod.set_cuenta(cuenta_id)
    return await RunPod.terminate_pod(pid)

# ══════════════════════════════════════════════════════════════
# HMAC
# ══════════════════════════════════════════════════════════════
def verify_signature(body: bytes, sig: str) -> bool:
    if not sig: return False
    expected = hmac.new(CALLBACK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)

# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.get("/api/health")
def health():
    return {"status":"ok", "time":datetime.now(timezone.utc).isoformat(),
            "motor": "2DGS", "imagen": RUNPOD_IMAGE,
            "runpod_configured": bool(RUNPOD_API_KEY),
            "runpod_cuenta2_configured": bool(RUNPOD_API_KEY_2),
            "r2_configured": bool(R2_ACCOUNT_ID and R2_ACCESS_KEY)}

@app.get("/api/runpod/cuentas")
def runpod_cuentas():
    return {"cuentas": _runpod_cuentas_disponibles()}

@app.get("/api/debug/gpus")
async def debug_gpus(cuenta: str = "1"):
    """Diagnóstico: muestra qué GPUs ve RunPod y cuáles coinciden con el ranking.
    Úsalo con ?cuenta=1 o ?cuenta=2 para probar cada cuenta."""
    RunPod.set_cuenta(cuenta)
    all_gpus = await RunPod.list_all_gpus()
    matched = []
    for label, req, forb in GPU_PREFERENCES:
        found = []
        for g in all_gpus:
            name = g.get("displayName","")
            if RunPod._matches(name, req, forb):
                found.append({"name":name, "id":g.get("id"), "vram":g.get("memoryInGb")})
        matched.append({"label":label, "matches":found})
    return {
        "cuenta_probada": cuenta,
        "total_gpus_que_ve_runpod": len(all_gpus),
        "todas": [{"name":g.get("displayName",""), "vram":g.get("memoryInGb",0),
                 "id":g.get("id")} for g in all_gpus],
        "coincidencias_ranking": matched,
    }

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), quality: str = Form("fast"),
                     cuenta_runpod: str = Form("1")):
    if quality not in ("fast","balanced","quality"):
        quality = "fast"
    if cuenta_runpod not in ("1", "2"):
        cuenta_runpod = "1"
    job_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    zip_key = f"uploads/{job_id}/input.zip"
    ply_key = f"results/{job_id}/mesh_2dgs.ply"
    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id,status,quality,created_at,updated_at,ply_key,
                              last_heartbeat,progress,message)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (job_id,"uploading",quality,now,now,ply_key,now,0.0,"Subiendo a R2"))
    job_update(job_id, runpod_cuenta=cuenta_runpod)
    try:
        r2_upload_file(file.file, zip_key)
    except Exception as e:
        job_update(job_id, status="error", error=f"R2 upload falló: {e}")
        raise HTTPException(500, f"R2 falló: {e}")
    # Variables que recibe el worker (URLs firmadas: sin credenciales R2 sueltas)
    env = {
        "TOUR_ID": job_id,
        "INPUT_URL": r2_get_url(zip_key, expires=7200),
        "UPLOAD_URL_PLY": r2_put_url(ply_key, expires=7200),
        "CALLBACK_URL": f"{BACKEND_URL}/api/internal/callback/{job_id}",
        "CALLBACK_SECRET": CALLBACK_SECRET,
        "QUALITY": quality,
    }
    try:
        pod_tagged, provider, gpu_label, gpu_displayname, disk = \
            await provision_gpu(job_id, env, cuenta_id=cuenta_runpod)
    except Exception as e:
        job_update(job_id, status="error", error=f"Sin GPU: {e}")
        raise HTTPException(503, f"Sin GPU disponible: {e}")
    job_update(job_id, status="processing",
               pod_id=pod_tagged,
               gpu_type=gpu_displayname,
               message=f"GPU {gpu_label} arrancando (~1 min)")
    return {"job_id":job_id, "status":"processing", "quality":quality,
            "pod_id":pod_tagged, "provider":provider, "cuenta_runpod":cuenta_runpod,
            "gpu_type":gpu_displayname, "gpu_label":gpu_label}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    j = job_get(job_id)
    if not j:
        raise HTTPException(404, "Job no encontrado")
    return {
        "job_id": j["id"], "status": j["status"], "quality": j.get("quality"),
        "progress": j.get("progress") or 0, "message": j.get("message") or "",
        "pod_id": j.get("pod_id"), "gpu_type": j.get("gpu_type"),
        "frames_used": j.get("frames_used"), "ply_mb": j.get("ply_mb"),
        "seconds": j.get("seconds"), "error": j.get("error"),
        "has_log": bool(j.get("worker_log")),
    }

@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str):
    j = job_get(job_id)
    if not j: raise HTTPException(404, "Job no encontrado")
    if j["status"] != "completed":
        raise HTTPException(400, f"Job no listo (estado: {j['status']})")
    return {
        "job_id": job_id,
        "ply_url": r2_get_url(j["ply_key"]),
        "ply_mb": j.get("ply_mb", 0),
    }

@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str):
    j = job_get(job_id)
    if not j: raise HTTPException(404, "Job no encontrado")
    header = (
        f"================================================\n"
        f"LOG DE RENDERIZADO 2DGS\n"
        f"Job: {job_id}\nEstado: {j['status']}\nCalidad: {j.get('quality')}\n"
        f"GPU: {j.get('gpu_type')}\nPod ID: {j.get('pod_id')}\n"
        f"Error: {j.get('error') or 'N/A'}\n"
        f"================================================\n\n"
    )
    return header + (j.get("worker_log") or "(Sin log)")

@app.post("/api/internal/callback/{job_id}")
async def worker_callback(job_id: str, request: Request,
                          x_signature: str = Header(default="")):
    body = await request.body()
    if not verify_signature(body, x_signature):
        raise HTTPException(401, "Firma HMAC inválida")
    try:
        payload = json.loads(body)
    except Exception:
        raise HTTPException(400, "JSON inválido")
    j = job_get(job_id)
    if not j: raise HTTPException(404, "Job no encontrado")
    cb_type = payload.get("type", "")
    now = datetime.now(timezone.utc).isoformat()
    if cb_type == "progress":
        campos = dict(progress=float(payload.get("progress", 0)),
                      message=payload.get("message", "")[:200],
                      last_heartbeat=now)
        log_parcial = payload.get("log")
        if log_parcial:
            campos["worker_log"] = log_parcial
        job_update(job_id, **campos)
        return {"ok":True}
    elif cb_type == "completed":
        log_text = payload.get("log", "") or "(El render terminó bien; sin log detallado)"
        job_update(job_id, status="completed", progress=1.0, message="Completado",
                   frames_used=payload.get("frames_used", 0),
                   ply_mb=payload.get("ply_mb", 0),
                   seconds=payload.get("seconds", 0),
                   worker_log=log_text, last_heartbeat=now)
        if j.get("pod_id"):
            await terminate_any(j["pod_id"], j.get("runpod_cuenta") or "1")
        return {"ok":True}
    elif cb_type == "error":
        log_text = payload.get("log", "") or payload.get("error_message", "")
        job_update(job_id, status="error",
                   error=payload.get("error_message","Error desconocido")[:500],
                   worker_log=log_text, last_heartbeat=now)
        if j.get("pod_id"):
            await terminate_any(j["pod_id"], j.get("runpod_cuenta") or "1")
        return {"ok":True}
    return {"ok":False, "reason":"tipo callback desconocido"}

# ══════════════════════════════════════════════════════════════
# WATCHDOG (IDÉNTICO a tu backend viejo)
# ══════════════════════════════════════════════════════════════
async def watchdog_loop():
    while True:
        try:
            await _watchdog_pass()
        except Exception as e:
            print(f"[watchdog] error: {e}")
        await asyncio.sleep(WATCHDOG_INTERVAL_SEC)

async def _watchdog_pass():
    now = datetime.now(timezone.utc)
    with get_db() as db:
        rows = db.execute("""
            SELECT id, pod_id, last_heartbeat, runpod_cuenta FROM jobs
            WHERE status='processing' AND pod_id IS NOT NULL AND pod_id != ''
        """).fetchall()
    for r in rows:
        hb = r["last_heartbeat"]
        if not hb: continue
        try:
            hb_dt = datetime.fromisoformat(hb.replace("Z","+00:00"))
            age_min = (now - hb_dt).total_seconds() / 60
        except Exception:
            continue
        if age_min > POD_HEARTBEAT_TIMEOUT_MIN:
            print(f"[watchdog] Job {r['id']} sin HB hace {age_min:.0f} min — matando pod")
            await terminate_any(r["pod_id"], r["runpod_cuenta"] if "runpod_cuenta" in r.keys() else "1")
            job_update(r["id"], status="error",
                       error=f"Sin heartbeat hace {age_min:.0f} min (timeout)")
    known_runpod = set()
    with get_db() as db:
        for r in db.execute("SELECT pod_id FROM jobs WHERE pod_id IS NOT NULL").fetchall():
            prov, pid = _split_pod_tag(r["pod_id"])
            if pid:
                known_runpod.add(pid)
    try:
        pods = await RunPod.list_my_pods()
    except Exception:
        pods = []
    for p in pods:
        pid = p.get("id")
        if pid and pid not in known_runpod:
            uptime = (p.get("runtime") or {}).get("uptimeInSeconds", 0)
            if uptime and uptime > POD_MAX_LIFETIME_MIN * 60:
                print(f"[watchdog] Pod huérfano {pid} con {uptime}s — terminando")
                await RunPod.terminate_pod(pid)

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(watchdog_loop())
    print(f"Backend Render-GS (2DGS) iniciado. Imagen={RUNPOD_IMAGE}, "
          f"RunPod={'OK' if RUNPOD_API_KEY else 'NO'}, R2={'OK' if R2_ACCOUNT_ID else 'NO'}")

# ══════════════════════════════════════════════════════════════
# HTML — página simple para subir el ZIP
# ══════════════════════════════════════════════════════════════
HTML_PAGE = """<!DOCTYPE html>
<html lang="es"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Render-GS — Prueba 2DGS</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:linear-gradient(135deg,#0f0f0f 0%,#16213e 100%);color:#eee;min-height:100vh;padding:20px}
.container{max-width:720px;margin:0 auto}
h1{text-align:center;font-size:26px;margin-bottom:8px;
  background:linear-gradient(90deg,#00D9FF,#7B61FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;color:#888;margin-bottom:24px;font-size:14px}
.card{background:#1a1a1a;border-radius:16px;padding:24px;margin-bottom:20px;border:1px solid #2a2a2a}
#dropzone{border:2px dashed #00D9FF;border-radius:12px;padding:48px 24px;text-align:center;
  cursor:pointer;transition:.2s;background:#1f1f1f}
#dropzone:hover,#dropzone.drag{background:#18222a;border-color:#7B61FF}
.icon{font-size:48px;margin-bottom:12px}
.text{font-size:16px;color:#ccc}.sub{font-size:13px;color:#777;margin-top:8px}
.file-info{background:#0f2a0f;border:1px solid #2a5a2a;border-radius:8px;padding:12px;margin-top:16px;display:none}
select,button{width:100%;padding:14px;border-radius:10px;border:none;font-size:15px;margin-top:12px;cursor:pointer}
select{background:#2a2a2a;color:#eee}
.btn-primary{background:linear-gradient(90deg,#00D9FF,#7B61FF);color:#fff;font-weight:bold;font-size:16px}
.btn-primary:disabled{opacity:.4;cursor:not-allowed}
.btn-download{background:#7B61FF;color:#fff;font-weight:bold}
.btn-success{background:#4CAF50;color:#fff;font-weight:bold}
#progress{display:none}
.log-box{background:#0a0a0a;border:1px solid #2a2a2a;border-radius:8px;padding:16px;
  font-family:monospace;font-size:12px;color:#9fef9f;height:280px;overflow-y:auto;
  white-space:pre-wrap;margin-top:16px}
.status{text-align:center;padding:16px;font-size:16px;font-weight:bold}
.status.processing{color:#00D9FF}.status.success{color:#4CAF50}.status.error{color:#E53935}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid #333;border-top-color:#00D9FF;
  border-radius:50%;animation:spin 1s linear infinite;vertical-align:middle;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.hidden{display:none !important}
.bar{height:4px;background:#222;border-radius:2px;margin-top:12px;overflow:hidden}
.bar-fill{height:100%;background:linear-gradient(90deg,#00D9FF,#7B61FF);transition:width .3s}
</style></head><body>
<div class="container">
  <h1>🧪 Render-GS — Prueba 2DGS</h1>
  <p class="subtitle">Sube el ZIP · se alquila sola la RTX 4090 (o la siguiente en la lista) · entrega malla .ply</p>
  <div class="card" id="upload-card">
    <div id="dropzone">
      <div class="icon">📦</div>
      <div class="text">Arrastra tu ZIP aquí o haz click</div>
      <div class="sub">.zip con fotos (mínimo 20)</div>
    </div>
    <input type="file" id="fileInput" accept=".zip" style="display:none">
    <div class="file-info" id="fileInfo"></div>
    <label style="display:block;margin-top:12px;font-size:13px;color:#aaa">Cuenta de RunPod</label>
    <select id="cuenta"></select>
    <button class="btn-primary" id="renderBtn" disabled>🚀 Iniciar Renderizado 2DGS</button>
  </div>
  <div class="card" id="progress">
    <div class="status processing" id="statusText"><span class="spinner"></span>Procesando...</div>
    <div class="bar"><div class="bar-fill" id="barFill" style="width:0%"></div></div>
    <div class="log-box" id="logBox">Iniciando...</div>
    <div id="resultActions" class="hidden">
      <button class="btn-success hidden" id="viewBtn">⬇️ Descargar malla (.ply)</button>
      <button class="btn-download hidden" id="logBtn">📄 Ver / descargar log</button>
      <button class="btn-primary" id="newBtn">🔄 Probar otro ZIP</button>
    </div>
  </div>
</div>
<script>
const dz=document.getElementById('dropzone'),fi=document.getElementById('fileInput');
const info=document.getElementById('fileInfo'),btn=document.getElementById('renderBtn');
const up=document.getElementById('upload-card'),csel=document.getElementById('cuenta');
const pr=document.getElementById('progress'),st=document.getElementById('statusText');
const lb=document.getElementById('logBox'),ra=document.getElementById('resultActions');
const vb=document.getElementById('viewBtn'),lgb=document.getElementById('logBtn');
const nb=document.getElementById('newBtn'),bf=document.getElementById('barFill');
let sel=null,jid=null,timer=null;
(async()=>{try{const r=await fetch('/api/runpod/cuentas');const d=await r.json();
  csel.innerHTML='';
  (d.cuentas||[]).forEach(c=>{const o=document.createElement('option');
    o.value=c.id;o.textContent='RunPod: '+c.nombre;csel.appendChild(o)});
  if(!csel.options.length){const o=document.createElement('option');
    o.value='1';o.textContent='RunPod: Cuenta 1';csel.appendChild(o)}
}catch(e){const o=document.createElement('option');
  o.value='1';o.textContent='RunPod: Cuenta 1';csel.appendChild(o)}})();
dz.onclick=()=>fi.click();
dz.ondragover=e=>{e.preventDefault();dz.classList.add('drag')};
dz.ondragleave=()=>dz.classList.remove('drag');
dz.ondrop=e=>{e.preventDefault();dz.classList.remove('drag');if(e.dataTransfer.files.length)hf(e.dataTransfer.files[0])};
fi.onchange=e=>{if(e.target.files.length)hf(e.target.files[0])};
function hf(f){if(!f.name.toLowerCase().endsWith('.zip')){alert('Sube un .zip');return}
  sel=f;info.style.display='block';info.textContent='✓ '+f.name+' ('+(f.size/1048576).toFixed(1)+' MB)';btn.disabled=false}
function addLog(m){lb.textContent+='\\n'+m;lb.scrollTop=lb.scrollHeight}
btn.onclick=async()=>{if(!sel)return;up.classList.add('hidden');pr.style.display='block';
  ra.classList.add('hidden');lb.textContent='Subiendo ZIP a R2...';st.innerHTML='<span class="spinner"></span>Subiendo...';
  const fd=new FormData();fd.append('file',sel);fd.append('quality','fast');
  fd.append('cuenta_runpod',csel.value||'1');
  try{const r=await fetch('/api/jobs',{method:'POST',body:fd});
    if(!r.ok){throw new Error('HTTP '+r.status+': '+await r.text())}
    const d=await r.json();jid=d.job_id;
    addLog('✓ Job '+jid+' creado');addLog('✓ GPU alquilada: '+d.gpu_type+' ('+d.gpu_label+')');
    addLog('Esperando arranque del pod... luego COLMAP + 2DGS');
    st.innerHTML='<span class="spinner"></span>Pod arrancando...';startPoll()
  }catch(e){addLog('❌ '+e.message);st.className='status error';st.textContent='❌ Error';showNew()}};
function startPoll(){
  timer=setInterval(async()=>{
    try{const r=await fetch('/api/jobs/'+jid);const j=await r.json();
      const p=Math.round((j.progress||0)*100);bf.style.width=p+'%';
      st.innerHTML='<span class="spinner"></span>'+(j.message||'Procesando')+' ('+p+'%)';
      if(j.status==='completed'){clearInterval(timer);
        addLog('');addLog('✅ MALLA LISTA');
        addLog('Frames: '+(j.frames_used||'?')+' · '+(j.ply_mb||'?')+' MB · '+(j.seconds||'?')+'s');
        st.className='status success';st.textContent='✅ ¡Completado!';bf.style.width='100%';
        ra.classList.remove('hidden');vb.classList.remove('hidden');
        vb.onclick=async()=>{const dr=await fetch('/api/jobs/'+jid+'/download');const dd=await dr.json();
          window.open(dd.ply_url,'_blank')};
        lgb.textContent='📄 Ver / descargar log';lgb.classList.remove('hidden');
        lgb.onclick=()=>window.open('/api/jobs/'+jid+'/log','_blank');showNew()
      }else if(j.status==='error'){clearInterval(timer);
        addLog('');addLog('❌ ERROR: '+(j.error||'sin detalle'));
        st.className='status error';st.textContent='❌ Falló';
        ra.classList.remove('hidden');vb.classList.add('hidden');
        lgb.textContent='📄 Descargar log del error';lgb.classList.remove('hidden');
        lgb.onclick=()=>window.open('/api/jobs/'+jid+'/log','_blank');showNew()
      }
    }catch(e){addLog('⚠ '+e.message)}
  },10000)}
function showNew(){nb.onclick=()=>location.reload()}
</script></body></html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
