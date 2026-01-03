# Promoting Dev (Sandbox) to Staging (Production) Plan

## Executive Summary

This plan outlines the steps to "Promote" your current working state on **Dev (Sandbox)** to the stable **Staging (Production)** environment and perform the final domain cutover (`app.khorium.ai` -> Staging). After execution, Staging will be your official stable Production environment.

---

## Current Infrastructure State (Discovered)

| Component | Dev (Sandbox) | Staging (Production) | Parity Gap |
|-----------|-----------------|---------|------------|
| ALB | webdev-alb (HTTP:80) | webdev-alb-stg (HTTP:80) | STOP: No HTTPS listener |
| CloudFront | E352AHA7L040MU | [x] None | STOP: Missing (Staging needs CF or HTTPS ALB) |
| SSL Cert (us-east-1) | *.khorium.ai [x] | (Shared) | [x] Ready for CF |
| SSL Cert (us-west-1) | *.khorium.ai [x] | (Available, not attached) | STOP: Needs HTTPS listener |
| RDS | khorium-webdev-db | khorium-staging-db (Empty) | STOP: No Production Data |
| S3 Frontend | muaz-mesh-web-dev | muaz-mesh-web-staging | STOP: Needs final build |
| S3 Assets | muaz-webdev-assets | (Same bucket?) | TBD (Isolation recommended) |
| EC2 | mesh-gen-web-prod-2 | mesh-gen-web-staging | [x] t3.micro parity |

---

## Phase 1: RDS Production Data Promotion

### 1.1 Overview

The goal is to promote a verified snapshot of your **Dev (Sandbox)** database to the **Staging (Production)** RDS instance. This ensures users, projects, and records move to the stable environment.

> [!CAUTION]
> Data Sensitivity: User passwords (hashed) and email addresses will be copied. Ensure staging access is restricted.

### 1.2 Pre-Migration Checklist

```bash
# Verify connectivity to both RDS instances
# From your local machine (ensure VPN/bastion if RDS is private)
psql -h khorium-webdev-db.cpw2scio00gm.us-west-1.rds.amazonaws.com -U postgres -d meshgen_db -c "SELECT count(*) FROM users;"
psql -h khorium-staging-db.cpw2scio00gm.us-west-1.rds.amazonaws.com -U postgres -d meshgen_db -c "SELECT 1;"
```

### 1.3 Option A: pg_dump / pg_restore (Recommended for <1GB DBs)

Estimated Time: 5-15 minutes depending on DB size.

```bash
# Step 1: SSH into the PRODUCTION EC2 instance (has network access to RDS)
ssh -i ~/.ssh/khorium-webdev-key-uswest1.pem ec2-user@54.183.252.115

# Step 2: Dump the production database
pg_dump -h khorium-webdev-db.cpw2scio00gm.us-west-1.rds.amazonaws.com \
        -U postgres \
        -d meshgen_db \
        -F c \
        -f /tmp/meshgen_prod_backup_$(date +%Y%m%d_%H%M%S).dump

# Step 3: Verify dump file size
ls -lh /tmp/meshgen_prod_backup_*.dump

# Step 4: Restore to STAGING database
# STOP: This will OVERWRITE existing staging data
pg_restore -h khorium-staging-db.cpw2scio00gm.us-west-1.rds.amazonaws.com \
           -U postgres \
           -d meshgen_db \
           -c \
           --if-exists \
           /tmp/meshgen_prod_backup_*.dump

# Step 5: Verify migration
psql -h khorium-staging-db.cpw2scio00gm.us-west-1.rds.amazonaws.com \
     -U postgres \
     -d meshgen_db \
     -c "SELECT COUNT(*) as users FROM users; SELECT COUNT(*) as projects FROM projects;"
```

### 1.4 Option B: AWS RDS Snapshot + Restore (Better for >1GB or minimal downtime)

```bash
# Step 1: Create manual snapshot of production DB
aws rds create-db-snapshot \
    --db-instance-identifier khorium-webdev-db \
    --db-snapshot-identifier khorium-webdev-snapshot-$(date +%Y%m%d) \
    --region us-west-1

# Step 2: Wait for snapshot to complete (~5-30 min depending on size)
aws rds wait db-snapshot-available \
    --db-snapshot-identifier khorium-webdev-snapshot-$(date +%Y%m%d) \
    --region us-west-1

# Step 3: Delete current staging DB (if you want to restore FROM snapshot)
# STOP: This is destructive. Only do this if staging DB is disposable.
aws rds delete-db-instance \
    --db-instance-identifier khorium-staging-db \
    --skip-final-snapshot \
    --region us-west-1

# Step 4: Restore staging DB from production snapshot
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier khorium-staging-db \
    --db-snapshot-identifier khorium-webdev-snapshot-$(date +%Y%m%d) \
    --db-instance-class db.t3.micro \
    --region us-west-1

# Step 5: Update security groups to match original staging config
aws rds modify-db-instance \
    --db-instance-identifier khorium-staging-db \
    --vpc-security-group-ids sg-01741701ab3597b82 \
    --region us-west-1 \
    --apply-immediately
```

### 1.5 Post-Migration: Update Staging .env

SSH into Staging EC2 and verify the database connection string:

```bash
ssh -i ~/.ssh/khorium-webdev-key-uswest1.pem ec2-user@3.101.132.150

# Check current .env
cat /home/ec2-user/backend/backend/.env | grep DATABASE

# If it points to production, update it:
# DATABASE_URL=postgresql://postgres:<password>@khorium-staging-db.cpw2scio00gm.us-west-1.rds.amazonaws.com:5432/meshgen_db

# Restart backend to pick up changes
sudo systemctl restart gunicorn
```

---

## Phase 2: S3 Asset Synchronization

### 2.1 Frontend Assets (React Build)

The staging branch auto-deploys to muaz-mesh-web-staging. If you want exact parity with production:

```bash
# Sync production frontend to staging bucket
aws s3 sync s3://muaz-mesh-web-dev s3://muaz-mesh-web-staging --delete
```

### 2.2 User-Uploaded Files (CAD, Meshes)

Based on config.py, production uses muaz-webdev-assets bucket. Check if staging uses the same or a separate bucket.

```bash
# List current staging assets
aws s3 ls s3://muaz-webdev-assets --recursive --summarize

# If staging should have its own copy (isolation):
# Create staging assets bucket
aws s3 mb s3://muaz-webdev-assets-staging --region us-west-1

# Sync all user data
aws s3 sync s3://muaz-webdev-assets s3://muaz-webdev-assets-staging

# Then update staging .env:
# S3_BUCKET_NAME=muaz-webdev-assets-staging
```

> [!WARNING]
> If staging shares muaz-webdev-assets with production, user file operations on staging will affect production files. This is likely not what you want for a true staging environment.

---

## Phase 3: CORS and SSL Configuration

### 3.1 Current CORS Configuration

From api_server.py, CORS origins are:
- Hardcoded defaults: localhost:5173, localhost:3000, 127.0.0.1:5173, S3 website URL
- Config: CORS_ORIGINS from .env

For Staging to accept app.khorium.ai requests after cutover:

```bash
# SSH into staging EC2
ssh -i ~/.ssh/khorium-webdev-key-uswest1.pem ec2-user@3.101.132.150

# Edit .env
nano /home/ec2-user/backend/backend/.env

# Add/Update CORS_ORIGINS line:
# CORS_ORIGINS=https://app.khorium.ai,http://muaz-mesh-web-staging.s3-website-us-west-1.amazonaws.com,http://localhost:5173

# Restart
sudo systemctl restart gunicorn
```

### 3.2 SSL Option A: Create CloudFront Distribution for Staging (Recommended)

This mirrors the production architecture (CloudFront -> ALB) and provides SSL termination.

```bash
# You already have a wildcard cert in us-east-1 (required for CloudFront):
# arn:aws:acm:us-east-1:571832839665:certificate/3461bd50-ee2c-4f96-9a5c-cf38f3c73c26

# Create CloudFront distribution via Console:
# 1. Go to CloudFront -> Create Distribution
# 2. Origin 1 (Frontend): muaz-mesh-web-staging.s3-website-us-west-1.amazonaws.com (HTTP only)
# 3. Origin 2 (Backend): webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com (HTTP only)
# 4. Behaviors:
#    - Default (*): Route to S3 origin
#    - /api/*: Route to ALB origin (with headers: Host, Authorization, Content-Type)
# 5. Alternate domain: staging.khorium.ai (or leave blank for testing)
# 6. SSL Certificate: Select *.khorium.ai from ACM
# 7. Create and note the distribution ID
```

**Programmatic Creation (JSON config):**

```json
{
  "CallerReference": "staging-distro-2026",
  "Aliases": {
    "Quantity": 1,
    "Items": ["staging.khorium.ai"]
  },
  "Origins": {
    "Quantity": 2,
    "Items": [
      {
        "Id": "S3-Frontend",
        "DomainName": "muaz-mesh-web-staging.s3-website-us-west-1.amazonaws.com",
        "CustomOriginConfig": {
          "HTTPPort": 80,
          "OriginProtocolPolicy": "http-only"
        }
      },
      {
        "Id": "ALB-Backend",
        "DomainName": "webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com",
        "CustomOriginConfig": {
          "HTTPPort": 80,
          "OriginProtocolPolicy": "http-only"
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-Frontend",
    "ViewerProtocolPolicy": "redirect-to-https"
  },
  "ViewerCertificate": {
    "ACMCertificateArn": "arn:aws:acm:us-east-1:571832839665:certificate/3461bd50-ee2c-4f96-9a5c-cf38f3c73c26",
    "SSLSupportMethod": "sni-only"
  },
  "Enabled": true
}
```

### 3.3 SSL Option B: Add HTTPS Listener to Staging ALB (Faster, Less Complete)

This adds SSL directly to the ALB but doesn't provide CDN caching or full CloudFront parity.

```bash
# Get the certificate ARN for us-west-1
CERT_ARN="arn:aws:acm:us-west-1:571832839665:certificate/9da4dfe8-75aa-49e7-8840-9f8349193fc4"

# Add HTTPS listener to staging ALB
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:us-west-1:571832839665:loadbalancer/app/webdev-alb-stg/f198da4d96a3d417 \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=$CERT_ARN \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-west-1:571832839665:targetgroup/webdev-app-tg-stg/cc3eef9c468b7069 \
    --region us-west-1

# Verify
aws elbv2 describe-listeners \
    --load-balancer-arn arn:aws:elasticloadbalancing:us-west-1:571832839665:loadbalancer/app/webdev-alb-stg/f198da4d96a3d417 \
    --region us-west-1
```

### 3.4 Frontend API URL Configuration

The staging frontend (muaz-mesh-web-staging) was built with VITE_API_URL_STAGING secret. Verify this points to the correct backend:

```bash
# Check what API URL was baked into the frontend build
# (This is in the compiled JS, search for "api" or the ALB hostname)
aws s3 cp s3://muaz-mesh-web-staging/assets/index-*.js - | grep -o 'https://[^"]*api[^"]*' | head -1

# If it points to production API, you need to rebuild:
# 1. Go to GitHub Secrets
# 2. Set VITE_API_URL_STAGING to the staging backend URL
#    - Option A (CloudFront): https://staging.khorium.ai
#    - Option B (Direct ALB): https://webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com
# 3. Re-trigger the staging workflow: git push origin staging
```

---

## Phase 4: Route53 Domain Cutover

### 4.1 Pre-Cutover Verification

```bash
# 1. Test staging health
curl -I http://webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com/api/health
# Expected: HTTP/1.1 200 OK

# 2. Test frontend loads
curl -s http://muaz-mesh-web-staging.s3-website-us-west-1.amazonaws.com | head -20

# 3. Test login (if CloudFront staging is set up with SSL)
curl -X POST https://staging.khorium.ai/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"test"}'
```

### 4.2 Cutover Procedure

> [!IMPORTANT]
> This changes where app.khorium.ai points. Users will be directed to staging.

**Option A: Point to Staging CloudFront (Recommended)**

```bash
# Get your Route53 hosted zone ID (from console if CLI doesn't show it)
# Then update the record

# Create change batch JSON
cat > /tmp/cutover-to-staging.json << 'EOF'
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "app.khorium.ai",
      "Type": "A",
      "AliasTarget": {
        "DNSName": "<STAGING_CLOUDFRONT_DOMAIN>.cloudfront.net",
        "HostedZoneId": "Z2FDTNDATAQYW2",
        "EvaluateTargetHealth": false
      }
    }
  }]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id <YOUR_ZONE_ID> \
    --change-batch file:///tmp/cutover-to-staging.json
```

**Option B: Point to Staging ALB Directly (Faster, HTTP-only unless you added HTTPS listener)**

```bash
cat > /tmp/cutover-to-staging-alb.json << 'EOF'
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "app.khorium.ai",
      "Type": "A",
      "AliasTarget": {
        "DNSName": "webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com",
        "HostedZoneId": "Z368ELLRRE2KJ0",
        "EvaluateTargetHealth": true
      }
    }
  }]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id <YOUR_ZONE_ID> \
    --change-batch file:///tmp/cutover-to-staging-alb.json
```

### 4.3 Rollback Procedure

```bash
# Point back to production CloudFront
cat > /tmp/rollback-to-prod.json << 'EOF'
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "app.khorium.ai",
      "Type": "A",
      "AliasTarget": {
        "DNSName": "d1234abcd.cloudfront.net",
        "HostedZoneId": "Z2FDTNDATAQYW2",
        "EvaluateTargetHealth": false
      }
    }
  }]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id <YOUR_ZONE_ID> \
    --change-batch file:///tmp/rollback-to-prod.json
```

---

## Phase 5: Verification and Smoke Tests

### 5.1 Post-Cutover Checks

```bash
# 1. DNS propagation
nslookup app.khorium.ai
# Should return staging ALB/CloudFront IP

# 2. SSL validation
curl -vI https://app.khorium.ai 2>&1 | grep -E "(SSL|subject|issuer)"

# 3. API health
curl https://app.khorium.ai/api/health

# 4. Login test (use a test account)
curl -X POST https://app.khorium.ai/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"your-test@email.com","password":"xxx"}'

# 5. Upload test (manual - use the web UI)
```

### 5.2 Monitoring

```bash
# Watch Gunicorn logs on staging
ssh -i ~/.ssh/khorium-webdev-key-uswest1.pem ec2-user@3.101.132.150
sudo journalctl -u gunicorn -f
```

---

## Summary: Minimum Viable Cutover Checklist

- [ ] RDS: pg_dump prod -> pg_restore staging
- [ ] S3 Frontend: Verify staging branch deployed or sync from prod
- [ ] S3 Assets: Decide shared vs isolated; sync if isolated
- [ ] .env Staging: DATABASE_URL -> staging RDS, CORS_ORIGINS -> include app.khorium.ai
- [ ] SSL: Either create staging CloudFront OR add HTTPS listener to ALB
- [ ] Frontend API URL: Verify VITE_API_URL points to staging backend
- [ ] Route53: Cutover app.khorium.ai to staging CloudFront/ALB
- [ ] Smoke Test: Health, Login, Upload, Download
