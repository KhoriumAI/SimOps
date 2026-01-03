# DNS Preparation & Staging Smoke Test Guide (GoDaddy Manual Mode)

Since you are managing DNS in GoDaddy, you will ignore AWS Route53 for DNS hosting and manually update your records.

---

## Part 1: Create `staging.khorium.ai` (The "Sandbox")

This allows you to test the new environment safely before pointing your main domain to it.

### 1. Get Your "Target" Hostname
You need to know where to point the staging address.
*   **If using CloudFront (Best)**: Go to AWS Console -> CloudFront. Find your Staging Distribution. Copy the **Domain Name** (e.g., `d1234abcd.cloudfront.net`).
*   **If using ALB directly**: Go to EC2 -> Load Balancers. Find `webdev-alb-stg`. Copy the **DNS Name** (e.g., `webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com`).

### 2. Add Record in GoDaddy
1.  Log in to GoDaddy and go to **DNS Management** for `khorium.ai`.
2.  Click **Add**.
3.  **Type**: `CNAME`
4.  **Name**: `staging`
5.  **Value**: Paste the CloudFront or ALB address from Step 1.
6.  **TTL**: 1/2 Hour (or default).
7.  **Save**.

> [!IMPORTANT]
> Wait a few minutes, then test: `https://staging.khorium.ai`. It should show your app.

---

## Part 2: Smoke Testing Checklist

Perform these tests on `https://staging.khorium.ai` BEFORE acting on the main domain.

### [ ] 1. Basic Availability
*   **Action**: Go to `https://staging.khorium.ai` in an Incognito/Private window.
*   **Pass**: The login page loads. No "Privacy Error" (SSL is good).

### [ ] 2. Database Connectivity (Login)
*   **Action**: Try to log in with an existing user account.
*   **Pass**: You are logged in and see the dashboard.
*   **Fail**: "Network Error" (Backend not reachable) or "Invalid Credentials" (DB migration failed).

### [ ] 3. Asset Loading (S3)
*   **Action**: Open a project or view a mesh.
*   **Pass**: The 3D viewer loads the model.
*   **Fail**: Infinite spinner or "File not found" (S3 bucket sync failed).

### [ ] 4. Upload Functionality (Write Test)
*   **Action**: Upload a small, dummy CAD file (e.g., `test.step`).
*   **Pass**: File uploads, appears in the list, and status changes (e.g., "Queued").

---

## Part 3: The Manual Cutover Workflow

1.  **Start the Promotion Script**:
    ```bash
    python scripts/promote_to_staging.py
    ```
2.  **Allow Data Sync**:
    *   Confirm "Yes" to Phase 1 (Database Snapshot & Restore).
    *   Confirm "Yes" to Phase 2 (Asset Sync).
3.  **PAUSE at the "STOP! DNS UPDATE REQUIRED" Prompt**:
    *   The script will pause and show you the target address.
    *   **Leave the terminal open.**
4.  **Smoke Test**:
    *   Perform the checks in Part 2.
    *   If they fail, type "NO" in the terminal to stop.
5.  **Update GoDaddy (The Real Cutover)**:
    *   Go back to GoDaddy DNS Management.
    *   Find your existing record for **`@`** (if A record) or **`app`** (if CNAME).
    *   **Update the Value** to match the Target Address shown in the terminal.
    *   **Save**.
6.  **Finalize**:
    *   Type **PROMOTE** (or "Yes") in the terminal to confirm you've done it.
    *   The script will verify and exit.
