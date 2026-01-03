# Deployment Guide

## Quick Deploy (Backend on EC2)

### Option 1: Manual SSM Deployment (Recommended)
Since GitHub Actions SSH keys may struggle with connectivity, the most reliable method is to use AWS Systems Manager (SSM) manually:

```powershell
# 1. Start Session
aws ssm start-session --target i-0bdb1e0eaa658cc7c

# 2. Switch to ec2-user
sudo su ec2-user

# 3. Deploy
cd ~/backend
git pull origin main
sudo systemctl restart gunicorn

# 4. Verify
curl localhost:3000/api/health
```

### Option 2: GitHub Actions (Automated)
The repository contains a `.github/workflows/deploy-backend.yml` that attempts to deploy via SSH.
**Prerequisites:**
- `EC2_HOST`, `EC2_SSH_KEY` secrets must be correct.
- Security Group must allow traffic from GitHub Actions IP (or open access) on Port 22.

**Verification command:**
```bash
# Check Gunicorn is on port 3000
sudo netstat -tlnp | grep 3000

# Test backend responds
curl http://localhost:3000/api/health
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

## Troubleshooting: Login/API Not Working

### Architecture Overview
```
User → CloudFront (app.khorium.ai) → ALB → EC2 (Gunicorn:3000) → Backend
```

### Quick Diagnostic Checklist

**1. Verify CloudFront Points to Correct ALB**
```powershell
# Get production ALB DNS
aws elbv2 describe-load-balancers --region us-west-1 --names webdev-alb --query "LoadBalancers[0].DNSName" --output text

# Check what CloudFront uses
aws cloudfront get-distribution-config --id E352AHA7L040MU --query "DistributionConfig.Origins.Items[?Id=='ALB-Backend'].DomainName" --output text
```

**These should match!** If CloudFront shows `webdev-alb-stg-*` (staging) instead of `webdev-alb-*` (production), update it:

```powershell
# Download config
aws cloudfront get-distribution-config --id E352AHA7L040MU > cf-temp.json

# Update origin (PowerShell)
$config = Get-Content cf-temp.json | ConvertFrom-Json
$etag = $config.ETag
$distConfig = $config.DistributionConfig
$distConfig.Origins.Items[0].DomainName = "webdev-alb-CORRECT-NAME.us-west-1.elb.amazonaws.com"
$distConfig | ConvertTo-Json -Depth 10 | Out-File -Encoding ascii cf-updated.json
aws cloudfront update-distribution --id E352AHA7L040MU --distribution-config file://cf-updated.json --if-match $etag
```

Wait 2-3 minutes for propagation.

**2. Check ALB Target Health**
```powershell
aws elbv2 describe-target-health --target-group-arn arn:aws:elasticloadbalancing:us-west-1:571832839665:targetgroup/webdev-app-tg/352fc88d69c940a2 --region us-west-1
```

If **unhealthy**, check:
- Gunicorn is running on port **3000**: `sudo netstat -tlnp | grep 3000`
- Security group allows port 3000 from ALB's security group
- Instance is in the correct target group

**3. Verify Gunicorn Port Configuration**

Gunicorn **must** listen on port **3000** to match the ALB target group:

```bash
# Check service file
cat /etc/systemd/system/gunicorn.service | grep -- "-b"
# Should show: -b 0.0.0.0:3000

# Test locally
curl http://localhost:3000/api/health
```

**4. Security Group Configuration**

EC2 instance needs inbound rules for:
- Port **3000** from ALB security group (`sg-0f2439ae16d06ddba`)
- Port **3000** from `0.0.0.0/0` (for health checks)

```powershell
# Verify rules
aws ec2 describe-security-groups --region us-west-1 --group-ids sg-0c9ab1036c0eab54d --query "SecurityGroups[0].IpPermissions[?ToPort==\`3000\`]"
```

**5. Watch for Incoming Connections**

On the server, monitor if requests are reaching the instance:

```bash
# Watch live logs
sudo journalctl -u gunicorn -f

# Watch network traffic
sudo tcpdump -i any port 3000 -n
```

If you see **nothing**, the ALB isn't connecting (go back to step 1-4).

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Login hangs, 504 error | CloudFront → wrong ALB | Update CloudFront origin |
| Target "unhealthy" | Port mismatch | Change Gunicorn to port 3000 |
| Timeout on health check | Security group blocks ALB | Add port 3000 from ALB SG |
| No traffic in tcpdump | Instance not in target group | Re-register instance |

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
