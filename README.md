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

## Normal User Flow

The user does not create a project manually. The app flow is:

```text
Enter topic -> get top 10 suggestions -> select one suggestion -> research job runs automatically
```

Use these endpoints for normal testing:

```text
POST /api/v1/research/suggestions
POST /api/v1/research/jobs/from-suggestion
GET  /api/v1/research/jobs/{job_id}
GET  /api/v1/research/jobs/{job_id}/events
GET  /api/v1/research/jobs/{job_id}/sources
GET  /api/v1/research/jobs/{job_id}/evidence
GET  /api/v1/research/jobs/{job_id}/report
POST /api/v1/research/jobs/{job_id}/review
POST /api/v1/research/jobs/{job_id}/report/regenerate
```

After report generation, the backend now runs a cost-free MVP verification pass:

```text
cited report -> citation guardrail auto-fix -> citation coverage check -> quality gate -> completed job
```

If the quality gate needs review, the frontend can trigger a stronger evidence pass:

```text
POST /api/v1/research/jobs/{job_id}/review
```

The frontend calls this automatically and renders the confidence result near the report citations:

```text
GET /api/v1/research/jobs/{job_id}/verification
```

Developer cost visibility is also available for Postman inspection:

```text
GET /api/v1/research/jobs/{job_id}/costs
```

`/api/v1/projects` is optional/internal for future saved workspaces, teams, and history.

Suggestion request body only needs a topic:

```json
{
  "topic": "electric vehicles in India"
}
```

The backend defaults are:

```json
{
  "audience": "general",
  "freshness": "latest"
}
```

Or run the helper script:

```powershell
.\scripts\run-dev.ps1
```

From Command Prompt or PowerShell, you can also run:

```powershell
.\run-dev.cmd
```
