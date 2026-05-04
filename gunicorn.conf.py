import os

_port = os.environ.get("PORT", "10000").strip() or "10000"
bind = f"0.0.0.0:{_port}"

worker_class = "uvicorn.workers.UvicornWorker"
workers = 1
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
