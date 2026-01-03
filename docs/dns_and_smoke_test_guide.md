# DNS Preparation & Staging Smoke Test Guide

## Part 1: DNS Provider Configuration (The "Switchboard")

Before AWS Route53 can manage your domain, you must tell your domain registrar (GoDaddy, Namecheap, etc.) to listen to AWS.

### 1. Get AWS Nameservers
1.  Log in to the **AWS Console**.
2.  Go to **Route53** -> **Hosted Zones**.
3.  Click on your domain name (e.g., `khorium.ai`).
4.  Look for the **NS (Name Server)** record type.
5.  You will see 4 values like:
    *   `ns-1234.awsdns-12.org`
    *   `ns-5678.awsdns-34.co.uk`
    *   `ns-9012.awsdns-56.com`
    *   `ns-3456.awsdns-78.net`

### 2. Update Registrar (GoDaddy/Namecheap/etc.)
1.  Log in to your **Domain Registrar**.
2.  Find **DNS Management** or **Nameservers** settings for your domain.
3.  Choose **"Custom Nameservers"** (not Default/Registrar).
4.  Enter the 4 AWS nameservers you found above.
5.  **Save**.

> [!IMPORTANT]
> **Propagation Time**: This can take 1-48 hours, but usually happens in minutes.
> **Verification**: Open a terminal and run `nslookup -type=ns khorium.ai`. It should show the AWS servers.

---

## Part 2: Create `staging.khorium.ai` (The "Sandbox")

This allows you to test the new environment safely without touching the live `app.khorium.ai`.

### 1. Create Staging CloudFront (Best Practice)
*   **Why**: Matches production architecture (SSL termination).
*   **How**:
    1.  Go to **CloudFront** console.
    2.  **Create Distribution**.
    3.  **Origin Domain**: Select your Staging S3 bucket website endpoint (`muaz-mesh-web-staging...`).
    4.  **Alternate Domain Names (CNAMEs)**: Add `staging.khorium.ai`.
    5.  **Custom SSL Certificate**: Select your `*.khorium.ai` certificate.
    6.  **Origins (Backend)**: Add a second origin pointing to `webdev-alb-stg-...` for `/api/*` path pattern.

### 2. Create DNS Record
1.  Go to **Route53**.
2.  **Create Record**.
3.  Name: `staging` (becomes `staging.khorium.ai`).
4.  Type: **A**.
5.  Toggle **Alias**: Yes.
6.  **Route Traffic To**: Alias to CloudFront Distribution -> Choose the one you just made.
7.  **Create Records**.

---

## Part 3: Smoke Testing Checklist

Perform these tests on `https://staging.khorium.ai` BEFORE running the final cutover script.

### [ ] 1. Basic Availability
*   **Action**: Go to `https://staging.khorium.ai` in an Incognito/Private window.
*   **Pass**: The login page loads. No "Privacy Error" (SSL is good).

### [ ] 2. Database Connectivity (Login)
*   **Action**: Try to log in with an existing user account.
*   **Context**: Since you copied the production DB, your *old* production login should work here.
*   **Pass**: You are logged in and see the dashboard.
*   **Fail**: "Network Error" (Backend not reachable) or "Invalid Credentials" (DB migration failed).

### [ ] 3. Asset Loading (S3)
*   **Action**: Open a project or view a mesh.
*   **Pass**: The 3D viewer loads the model.
*   **Fail**: Infinite spinner or "File not found" (S3 bucket sync failed).

### [ ] 4. Upload Functionality (Write Test)
*   **Action**: Upload a small, dummy CAD file (e.g., `test.step`).
*   **Pass**: File uploads, appears in the list, and status changes (e.g., "Queued").
*   **Note**: If Staging is ISOLATED, this will not appear in Production. This is good!

---

## Part 4: The Final Cutover Workflow

 The efficient way to do this is to use the script's built-in pause.

1.  **Start the Promotion Script**:
    ```bash
    python scripts/promote_to_staging.py
    ```
2.  **Allow Data Sync**:
    *   Confirm "Yes" to Phase 1 (Database Snapshot & Restore).
    *   Confirm "Yes" to Phase 2 (Asset Sync).
3.  **PAUSE at the Cutover Prompt**:
    *   The script will stop and ask: `🚨 CRITICAL SAFETY CHECK 🚨 ... Point app.khorium.ai to Staging ALB?`
    *   **DO NOT TYPE 'PROMOTE' YET.**
4.  **Perform Smoke Tests (Part 3)**:
    *   While the script is waiting, go to `https://staging.khorium.ai` and do your checks.
5.  **Finalize**:
    *   If tests pass: Go back to the terminal, type **PROMOTE**, and hit Enter.
    *   If tests fail: Type "NO" to abort the DNS change. Your data is staged, but traffic is still on the old production.
