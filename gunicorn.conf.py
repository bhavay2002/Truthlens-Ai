import os
import sys

# Read port — Render always sets $PORT; fall back to 10000
_port = os.environ.get("PORT", "10000").strip() or "10000"
bind = f"0.0.0.0:{_port}"

# ASGI worker required for FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

workers = 1
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Surface startup errors immediately instead of silent retry loops
preload_app = True
