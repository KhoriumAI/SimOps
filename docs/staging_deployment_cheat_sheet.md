# Staging (Production) Environment Deployment Cheat Sheet

**Goal:** Promote stable code from Sandbox (Dev) to Production (Staging).  
**Last Updated:** 2026-01-02

---

## Wiki Entry / Quick Reference (Copy-Paste)

**To Promote to Production (Staging):** `git push origin staging` (Frontend only; see Backend Manual steps below)

**Stable Production URL:** http://muaz-mesh-web-staging.s3-website-us-west-1.amazonaws.com

**Final Cutover:** "Go to Route53 -> Hosted Zones -> Edit Record 'app' -> Change Alias Target to `webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com`."

---

## EMERGENCY DEPLOYMENT PROCEDURES

### 1. Frontend Hotfix (Automated)
```bash
git checkout staging
git merge main  # or cherry-pick specific commit
git push origin staging
```
**Effect:** Auto-deploys via GitHub Actions to muaz-mesh-web-staging S3 bucket.

### 2. Backend Hotfix (Manual SSH)
> [!WARNING]
> STOP: No GitHub Action exists for backend staging deployment.

```bash
# 1. Connect to Staging EC2
ssh -i ~/.ssh/khorium-webdev-key-uswest1.pem ec2-user@3.101.132.150

# 2. Deploy Code
cd /home/ec2-user/backend
git fetch origin
git checkout staging
git pull origin staging

# 3. Restart Service
sudo systemctl restart gunicorn

# 4. Verify Health
curl http://localhost:3000/api/health
# Expected: {"status": "healthy", ...}
```

---

## DATABASE SAFETY AUDIT: PASSED (ISOLATED)

> [!TIP]
> Staging Database is ISOLATED from Production.

**Finding:**
- Production DB: khorium-webdev-db (us-west-1c, gp3)
- Staging DB: khorium-staging-db (us-west-1c, gp2)

**Implications:**
- [x] Safe to perform destructive tests on Staging.
- [x] Safe to wipe/reset Staging DB.
- Note: Ensure Staging EC2 .env points to khorium-staging-db.cpw2scio00gm.us-west-1.rds.amazonaws.com.

---

## INFRASTRUCTURE MAP

| Resource | Production | Staging | Status |
|----------|-----------|---------|--------|
| Frontend URL | app.khorium.ai | [Link](http://muaz-mesh-web-staging.s3-website-us-west-1.amazonaws.com) | [x] Parity |
| Backend URL | api.khorium.ai (via CF) | webdev-alb-stg-699722072... | [x] Parity |
| EC2 Instance | mesh-gen-web-prod-2 | mesh-gen-web-staging | [x] Parity (t3.micro) |
| Database | khorium-webdev-db | khorium-staging-db | [x] ISOLATED |
| Deployment | Auto (GitHub Action) | Mixed (Frontend Auto / Backend Manual) | STOP: Gap |

---

## ROUTE53 CUTOVER (Emergency Failover)

### When to Use
- Production EC2/ALB Failure
- "Switch to Staging" required immediately

### Procedure
1. Go to **Route53** -> **Hosted Zones** -> khorium.ai
2. Select record: app (A Record)
3. Click **Edit Record**
4. Toggle **Route traffic to** -> **Alias to Application and Classic Load Balancer**
5. Region: us-west-1
6. Choose Load Balancer: webdev-alb-stg (Search for webdev)
   - Target DNS: webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com
   - Target Zone ID: Z368ELLRRE2KJ0
7. Click **Save**

**Propagation:** ~60 seconds.
