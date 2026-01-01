# Deployment Guide

## Quick Deploy (EC2)

```bash
cd /home/ubuntu/MeshPackageLean
git pull origin main
sudo systemctl restart gunicorn
sudo systemctl status gunicorn
```bash
cd /home/ubuntu/MeshPackageLean
git pull origin main
sudo systemctl restart gunicorn
sudo systemctl status gunicorn
```

## Frontend Deploy (Windows -> S3)

To update the frontend (app.khorium.ai), you must run the deployment script from your local machine:

```powershell
.\deploy.ps1
```

This builds the React app and syncs it to the S3 bucket.
**Note:** You may need to invalidate the CloudFront cache if changes don't appear immediately.


## Why Restart?

Code lives in two places:

| Location | Purpose |
|----------|---------|
| **Disk** | Storage (`/home/ubuntu/MeshPackageLean/`) |
| **RAM**  | Execution (Gunicorn loads code into memory) |

- `git pull` → Updates files on **disk**
- `systemctl restart` → Reloads code from disk into **RAM**

**Benefits of RAM execution:**
- ~100,000x faster than disk access
- Code compiled once at startup
- Multiple workers handle parallel requests

**Tradeoff:** Requires restart to pick up code changes.

## CORS Configuration

CORS origins are set via environment variable in `/home/ubuntu/MeshPackageLean/backend/.env`:

```bash
CORS_ORIGINS=https://your-frontend.com,https://another-domain.com
```

After changing, restart Gunicorn:
```bash
sudo systemctl restart gunicorn
sudo systemctl restart gunicorn
```

## Environment Configuration

To enable Slack notifications for feedback, add the webhook URL to `/home/ubuntu/MeshPackageLean/backend/.env`:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Then restart Gunicorn:
```bash
sudo systemctl restart gunicorn
```

**Default allowed origins** (hardcoded fallbacks in `api_server.py`):
- `http://localhost:5173` (Vite dev)
- `http://localhost:3000`
- Production domains

## Nginx (One-Time Setup)

File upload limit is set in `/etc/nginx/nginx.conf`:
```nginx
client_max_body_size 1G;
```

After changing:
```bash
sudo nginx -t && sudo systemctl reload nginx
```

## Rollback

```bash
git log --oneline -5           # Find previous commit
git checkout <commit-hash>     # Checkout that commit
sudo systemctl restart gunicorn
```

## Lessons Learned: Ghost Deployment Incident

**What happened:** Server was running code from RAM that didn't match disk files. Restarting Gunicorn loaded a skeleton version, breaking login/upload functionality.

**Root causes:**
- Uncommitted code running in production
- Directory structure drift (`backend/backend/` vs `backend/`)
- Nginx blocking large uploads (413 error)

**Prevention:**
1. Never edit files directly on server
2. Always commit to Git before deploying
3. Use `git pull` - never copy-paste files
4. Verify after restart: `curl http://localhost:5000/api/health`

## Golden Rule

**No code goes to production unless it has a Git Commit Hash.**

---

## Job Logs (New)

Every import and mesh operation now generates a unique Job ID (e.g., `IMP-0101-ABCD` or `MSH-0101-EFGH`). These are logged as structured JSON-lines for easy searching and debugging.

### Log Location
Logs are stored in the backend folder and rotate monthly:
`/home/ubuntu/MeshPackageLean/backend/logs/jobs_YYYY-MM.jsonl`

### Quick Troubleshooting

**Watch live logs:**
```bash
tail -f /home/ubuntu/MeshPackageLean/backend/logs/jobs_$(date +%Y-%m).jsonl
```

**Search for a specific Job ID provided by a user:**
```bash
grep "MSH-0101-ABCD" /home/ubuntu/MeshPackageLean/backend/logs/jobs_*.jsonl
```

**Find all failed jobs today:**
```bash
grep "\"status\": \"error\"" /home/ubuntu/MeshPackageLean/backend/logs/jobs_$(date +%Y-%m).jsonl | grep $(date +%m%d)
```

### Admin API
You can also query logs via API:
`GET /api/admin/logs?job_id=MSH-...` (Requires admin JWT)
