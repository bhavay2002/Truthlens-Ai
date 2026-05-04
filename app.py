import os
import sys
import traceback

try:
    from api.main import app
except Exception:
    # If the full app fails to import, expose a minimal app that
    # reports the error on every endpoint so the port opens and
    # the traceback is visible in Render logs.
    _tb = traceback.format_exc()
    print(f"[app.py] STARTUP ERROR — falling back to error app:\n{_tb}", file=sys.stderr)

    from fastapi import FastAPI

    app = FastAPI(title="TruthLens — startup error")

    @app.get("/")
    @app.get("/health")
    async def startup_error():
        return {"status": "error", "detail": _tb}
