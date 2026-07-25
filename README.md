# AI Deep Research Backend

FastAPI backend for the AI Deep Research Platform.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

API docs:

```text
http://127.0.0.1:8001/docs
```

Or run the helper script:

```powershell
.\scripts\run-dev.ps1
```

From Command Prompt or PowerShell, you can also run:

```powershell
.\run-dev.cmd
```
