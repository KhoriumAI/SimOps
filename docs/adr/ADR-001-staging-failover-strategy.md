# ADR-001: Environment Topology and Stability Strategy

**Status:** Proposed  
**Date:** 2026-01-02  
**Deciders:** CTO, Lead Architect  
**Technical Story:** Define the role of Staging as the stable Production environment and Dev as the experimental Sandbox.

---

## Context and Problem Statement

Currently, the Khorium MeshGen platform live domain (`app.khorium.ai`) points to the **Dev** environment. This is risky because Dev is used for experimental features and can break frequently.

The **Staging** environment has been provisioned to serve as the stable **Production** home for the application. We need to formalize:

1. **Environmental Roles:** Staging = Stable Production; Dev = Experimental Sandbox.
2. **Promotion Mechanism:** How code and data move from Sandbox (Dev) to Production (Staging).
3. **Data Parity:** How Production (Staging) stays isolated but updated with verified data.

---

## Decision Drivers

- **CTO Independence:** The CTO must be able to execute failover without needing DevOps or engineering support.
- **Cost Efficiency:** The staging environment should not incur significant costs when idle.
- **Data Integrity:** User data (projects, mesh results) must not be lost during failover.
- **Simplicity:** The failover procedure must be executable in <5 minutes with documented commands.

---

## Considered Options

### Option 1: Cold Standby (Current State)

**Description:** Staging exists but has no data. Failover requires full migration before cutover.

| Metric | Value |
|--------|-------|
| **RTO** | 30-60 minutes |
| **RPO** | Depends on last `pg_dump` |
| **Cost** | ~$50/month (idle EC2 + RDS) |
| **Complexity** | High (full migration during incident) |

**Pros:**
- Lowest cost
- No ongoing sync overhead

**Cons:**
- Failover is slow and error-prone under pressure
- Requires SSH access and database expertise during incident

---

### Option 2: Warm Standby with Periodic Sync (Recommended)

**Description:** Staging maintains near-real-time data parity via scheduled `pg_dump` (every 6 hours) and shared S3 bucket. Failover requires only DNS cutover.

| Metric | Value |
|--------|-------|
| **RTO** | 5-10 minutes |
| **RPO** | <=6 hours (last sync) |
| **Cost** | ~$60/month (same + cron job) |
| **Complexity** | Low (DNS change only) |

**Pros:**
- Fast failover (DNS change only)
- Minimal data loss (6-hour window)
- CTO can execute without engineering

**Cons:**
- Requires cron job setup for periodic sync
- 6-hour RPO may lose some user data

---

### Option 3: Hot Standby with RDS Read Replica

**Description:** Staging RDS is a read replica of Production. On failover, promote replica to primary.

| Metric | Value |
|--------|-------|
| **RTO** | 2-5 minutes |
| **RPO** | ~0 (real-time replication) |
| **Cost** | ~$100/month (replica instance) |
| **Complexity** | Medium (replica promotion) |

**Pros:**
- Near-zero data loss
- Fastest failover

**Cons:**
- Higher cost
- Replica promotion requires AWS CLI or Console access
- Read replica cannot serve writes until promoted

---

### Option 4: Multi-AZ + Auto Scaling (Enterprise)

**Description:** Full HA architecture with multi-AZ RDS, auto-scaling EC2, and health-check-based automatic failover.

| Metric | Value |
|--------|-------|
| **RTO** | <1 minute (automatic) |
| **RPO** | 0 |
| **Cost** | ~$300-500/month |
| **Complexity** | High (infrastructure redesign) |

**Pros:**
- Zero-touch failover
- Enterprise-grade reliability

**Cons:**
- Significant cost increase
- Over-engineered for current scale

---

## Decision Outcome

**Chosen Option:** **Staging as Permanent Stable Production**

### Rationale

1. **Safety:** By moving the main domain to Staging, experimental work on Dev can continue without causing downtime for users.
2. **Isolation:** Production (Staging) uses its own RDS, preventing experimental database schema changes from corrupting live user data.
3. **Promotion Workflow:** Dev serves as the staging ground for features; once verified, they are "promoted" to Staging (Production).

### Implementation Requirements

1. **Promotion Sync:** A controlled process (manual or automated) to move verified data and code from Dev to Staging.
2. **Production DNS:** The `app.khorium.ai` domain must be pointed to the Staging ALB.
3. **SSL at Edge:** Staging must terminate SSL (`*.khorium.ai`) via ALB or CloudFront.
4. **Environment Variables:** Staging config must be finalized for Production (RDS endpoints, S3 buckets, CORS).

---

## Consequences

### Positive

- CTO can failover without engineering support
- Minimal cost increase
- Clear RPO/RTO expectations

### Negative

- Up to 6 hours of data loss possible
- Manual sync requires monitoring (cron failure = stale staging)
- Two environments to maintain

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Cron job fails silently | Add CloudWatch alarm on sync failure |
| Staging EC2 goes unhealthy | Weekly health check (manual or automated) |
| Database schema drift | Always run migrations on BOTH environments |
| Secrets mismatch | Document all secrets; verify quarterly |

---

## Operational Runbooks

### Runbook 1: Emergency Failover to Staging

**Trigger:** Production is unresponsive after 5-minute grace period.

**Pre-Requisites Met:**
- [ ] Staging database was synced within last 6 hours
- [ ] Staging EC2 is running (`aws ec2 describe-instances`)
- [ ] Staging Gunicorn is healthy (`curl http://3.101.132.150:3000/api/health`)

**Procedure:**

```bash
# 1. Verify staging health
curl -s http://webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com/api/health

# 2. Execute DNS cutover (Console method)
# Route53 -> Hosted Zones -> khorium.ai -> Edit 'app' record
# Change Alias Target to: webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com
# Zone ID: Z368ELLRRE2KJ0

# 3. Wait 60 seconds for DNS propagation
sleep 60

# 4. Verify cutover
nslookup app.khorium.ai
curl -s https://app.khorium.ai/api/health

# 5. Notify team
# Post in Slack: "STOP: FAILOVER EXECUTED: app.khorium.ai now points to Staging"
```

**Estimated Time:** 5 minutes

---

### Runbook 2: Rollback to Production

**Trigger:** Production is restored and verified healthy.

**Procedure:**

```bash
# 1. Verify production health
curl -s http://webdev-alb-1882895883.us-west-1.elb.amazonaws.com/api/health

# 2. Sync any new data from Staging back to Production (if applicable)
# This is complex; may require manual merge of database records created during outage

# 3. Execute DNS rollback
# Route53 -> Hosted Zones -> khorium.ai -> Edit 'app' record
# Change Alias Target back to: [Production CloudFront] or [Production ALB]

# 4. Notify team
# Post in Slack: "[x] ROLLBACK COMPLETE: app.khorium.ai restored to Production"
```

---

### Runbook 3: Periodic Database Sync (Cron)

**Schedule:** Every 6 hours (0 */6 * * *)

**Location:** Production EC2 crontab

**Script:** `/home/ec2-user/scripts/sync_to_staging.sh`

```bash
#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DUMP_FILE="/tmp/meshgen_backup_${TIMESTAMP}.dump"
S3_BACKUP="s3://muaz-webdev-backups/db/${TIMESTAMP}.dump"

# 1. Dump production database
pg_dump -h khorium-webdev-db.cpw2scio00gm.us-west-1.rds.amazonaws.com \
        -U postgres \
        -d meshgen_db \
        -F c \
        -f "$DUMP_FILE"

# 2. Upload to S3 for audit trail
aws s3 cp "$DUMP_FILE" "$S3_BACKUP"

# 3. Restore to staging
pg_restore -h khorium-staging-db.cpw2scio00gm.us-west-1.rds.amazonaws.com \
           -U postgres \
           -d meshgen_db \
           -c \
           --if-exists \
           "$DUMP_FILE" 2>&1 || true  # Ignore non-fatal errors

# 4. Cleanup
rm -f "$DUMP_FILE"

# 5. Log success
echo "[$(date)] Staging sync completed successfully" >> /var/log/staging_sync.log
```

**Monitoring:**

```bash
# Add to CloudWatch via awslogs agent
# Or simple check:
tail -1 /var/log/staging_sync.log | grep "$(date +%Y-%m-%d)"
```

---

## Appendix: Infrastructure Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │ CloudFront  │────▶│  webdev-alb │────▶│ mesh-gen-web-prod-2 │   │
│  │ E352AHA7L  │     │  (HTTP:80)  │     │     (t3.micro)      │   │
│  └─────────────┘     └─────────────┘     └─────────────────────┘   │
│         │                                         │                 │
│         ▼                                         ▼                 │
│  ┌─────────────┐                         ┌─────────────────────┐   │
│  │ S3 Frontend │                         │  khorium-webdev-db  │   │
│  │ (web-dev)   │                         │   (RDS Postgres)    │   │
│  └─────────────┘                         └─────────────────────┘   │
│                                                   │                 │
│                                          ┌────────┴───────┐        │
│                                          │ pg_dump (6hr)  │        │
│                                          └────────┬───────┘        │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          STAGING                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │ CloudFront  │────▶│webdev-alb-stg────▶│ mesh-gen-web-staging│   │
│  │ (Optional)  │     │  (HTTP:80)  │     │     (t3.micro)      │   │
│  └─────────────┘     └─────────────┘     └─────────────────────┘   │
│         │                                         │                 │
│         ▼                                         ▼                 │
│  ┌─────────────┐                         ┌─────────────────────┐   │
│  │ S3 Frontend │                         │  khorium-staging-db │   │
│  │ (staging)   │                         │   (RDS Postgres)    │   │
│  └─────────────┘                         └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │      Route53        │
                    │   app.khorium.ai    │
                    │  (Alias to Prod CF) │
                    │                     │
                    │  [FAILOVER: Switch  │
                    │   to Staging ALB]   │
                    └─────────────────────┘
```

---

## Related Documents

- [Staging Deployment Cheat Sheet](../staging_deployment_cheat_sheet.md)
- [Migration and Cutover Plan](../migration_and_cutover_plan.md)
- [DEPLOYMENT.md](../AWS_DEPLOYMENT.md)
