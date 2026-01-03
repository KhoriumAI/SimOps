# CloudFront Mixed Environment Routing
*Status:* Accepted (Issue Identified)  
*Date:* 2026-01-02  
*Tags:* #infrastructure, #deployment, #cloudfront, #security

## 1. Context & Problem Statement

The production domain `app.khorium.ai` is served by a CloudFront distribution (E352AHA7L040MU) that is currently misconfigured, resulting in a mixed environment where frontend assets come from one environment (DEV) and backend API requests route to another (STAGING).

**The Constraint:**
- Users accessing `app.khorium.ai` were seeing outdated frontend UI despite recent deployments
- Users were unexpectedly logged in as "muaz" instead of their own accounts
- Frontend code changes deployed to DEV S3 bucket were not appearing to users

**The Discovery:**
CloudFront origin configuration shows:
- ✅ **Frontend Origin**: `muaz-mesh-web-dev.s3-website-us-west-1.amazonaws.com` (DEV environment)
- ❌ **Backend Origin**: `webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com` (STAGING ALB)

**The Goal:**
Ensure environment consistency - either both frontend and backend should be DEV, or both should be STAGING.

## 2. Technical Decision

**Immediate Actions Taken:**
1. ✅ Deployed updated frontend to DEV S3 bucket (`muaz-mesh-web-dev`)
2. ✅ Created CloudFront cache invalidation (ID: `I2IY79HV0S8YSUWCJNF9C71HWX`) to clear stale cached files
3. 📝 Documented this issue as ADR-0013

**Required Fix (Not Yet Implemented):**
Update CloudFront distribution to route `/api/*` and `/auth/*` to the **DEV ALB** instead of STAGING:

```json
// Current (Incorrect)
{
  "Id": "ALB-Backend",
  "DomainName": "webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com"  // STAGING
}

// Should be (Correct)
{
  "Id": "ALB-Backend", 
  "DomainName": "webdev-alb-XXXXX.us-west-1.elb.amazonaws.com"  // DEV ALB
}
```

**Dependencies:**
- CloudFront distribution update requires ETag from current config
- May require 2-5 minute propagation time after update
- Requires identifying the correct DEV ALB DNS name

## 3. Infrastructure & Security Implications

**Data Isolation:**
- ❌ **BROKEN**: Frontend loads from DEV, backend connects to STAGING database
- Users see STAGING user accounts (e.g., "muaz@khorium.ai") despite accessing what they believe is DEV
- Authentication tokens are issued against STAGING RDS instance
- Project uploads, mesh results, and user activity logs write to STAGING database

**Cache Behavior:**
- CloudFront was aggressively caching old frontend files
- Code deployments to S3 were not visible without explicit cache invalidation
- Default cache policy (ID: `658327ea-f89d-4fab-a63d-7e88639e58f6`) may need adjustment

**Security:**
- No actual security breach occurred
- The "logged in as muaz" issue is a symptom of environment mismatch, not unauthorized access
- However, this configuration creates confusion about which environment users are interacting with

## 4. Performance Trade-offs

**CloudFront Cache Invalidation:**
- **Cost**: AWS charges $0.005 per invalidation path after first 1,000 paths/month
- **Time**: Invalidations take 1-5 minutes to propagate globally
- **Alternative**: Version-based asset names (e.g., `index-ClNzKR6b.js`) provide automatic cache busting

**Mixed Environment Impact:**
- **Latency**: No measurable latency impact (both DEV and STAGING are in us-west-1)
- **State Confusion**: Developers may make changes to DEV expecting to see results, but authentication and data operations happen against STAGING
- **Testing Integrity**: Cannot reliably test integrated frontend + backend changes in isolation

## 5. Verification Plan

**Sanity Check:**
```bash
# 1. Verify frontend is served from DEV bucket
curl -I https://app.khorium.ai/ | grep -i x-cache

# 2. Verify API routes to correct backend
curl -I https://app.khorium.ai/api/health

# 3. Test login against expected database
curl -X POST https://app.khorium.ai/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"mark@khorium.ai","password":"xxx"}'
# Expected: Should authenticate against DEV database users
```

**Regression:**
- After CloudFront update, verify cache invalidation is no longer required for routine deployments
- Confirm that `deploy.ps1` successfully updates frontend without manual CloudFront intervention
- Validate that authentication reflects the correct environment's user database

## 6. Related Files

- **CloudFront Config**: `distribution_config.json` (lines 17, 44)
- **Deployment Script**: `deploy.ps1` (deploys to DEV bucket only)
- **Migration Plan**: `docs/migration_and_cutover_plan.md` (explains DEV vs STAGING architecture)
- **Deployment Docs**: `DEPLOYMENT.md` (frontend deployment process)

## 7. Next Steps

1. **Identify DEV ALB DNS name** via AWS console or CLI
2. **Update CloudFront distribution** to point backend origin to DEV ALB
3. **Test end-to-end** after CloudFront propagation
4. **Consider**: Should `app.khorium.ai` point to DEV permanently, or is STAGING the intended production environment?
5. **Document decision** in infrastructure docs about which environment is "production"
