# Validation Guide for MESH-104 Task

This guide helps you capture the required screenshots and validate the implementation.

## Prerequisites

1. Backend server running: `cd backend && python3 api_server.py`
2. Database migrated: `alembic upgrade head` (already done)
3. At least one user account created
4. At least one project uploaded

## Screenshot 1: JobUsage Table with Data

### Option A: Using SQLite CLI

```bash
cd backend
sqlite3 instance/mesh_app.db

# Run these commands in sqlite3:
.headers on
.mode column
SELECT * FROM job_usage ORDER BY started_at DESC LIMIT 10;

# Or prettier format:
.mode markdown
SELECT 
    id, 
    user_id, 
    job_id, 
    job_type, 
    status, 
    started_at,
    compute_backend
FROM job_usage 
ORDER BY started_at DESC 
LIMIT 10;
```

**Take screenshot** of the output showing populated JobUsage records.

### Option B: Using Test Script

```bash
cd backend
python3 test_rate_limiting.py <your_email> <your_password> 2
```

The script will display the JobUsage table contents. **Take screenshot** of the output.

### Option C: Using Python Script

```python
import sqlite3
from pathlib import Path

db_path = Path("backend/instance/mesh_app.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

cursor.execute("SELECT * FROM job_usage ORDER BY started_at DESC LIMIT 10")
rows = cursor.fetchall()

print("JobUsage Table Contents:")
for row in rows:
    print(row)

conn.close()
```

## Screenshot 2: HTTP 429 Error Response

### Method 1: Using Test Script (Recommended)

1. **Set low quota for testing:**
   ```bash
   export DEFAULT_JOB_QUOTA=2
   cd backend
   python3 api_server.py
   ```

2. **In another terminal, run the test script:**
   ```bash
   cd backend
   python3 test_rate_limiting.py <your_email> <your_password> 2
   ```

3. **The script will:**
   - Generate 2 jobs successfully
   - Attempt a 3rd job
   - Show HTTP 429 response with error details

4. **Take screenshot** of the HTTP 429 response output.

### Method 2: Using curl

```bash
# Set quota to 2
export DEFAULT_JOB_QUOTA=2

# Login and get token
TOKEN=$(curl -s -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"your@email.com","password":"yourpassword"}' \
  | jq -r '.access_token')

# Generate first job (should succeed)
curl -X POST http://localhost:5000/api/projects/<project_id>/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mesh_strategy":"Tet (Fast)"}'

# Generate second job (should succeed)
curl -X POST http://localhost:5000/api/projects/<project_id>/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mesh_strategy":"Tet (Fast)"}'

# Generate third job (should return 429)
curl -v -X POST http://localhost:5000/api/projects/<project_id>/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mesh_strategy":"Tet (Fast)"}'
```

**Take screenshot** of the HTTP 429 response (status code and error message).

### Method 3: Using Browser DevTools

1. Open frontend at `http://localhost:3000`
2. Login
3. Set `DEFAULT_JOB_QUOTA=2` in backend environment
4. Generate 3 mesh jobs quickly
5. Open browser DevTools → Network tab
6. Find the 3rd request that returned 429
7. **Take screenshot** showing:
   - Status: 429
   - Response body with error message

## Expected HTTP 429 Response

```json
{
  "error": "Daily job quota exceeded",
  "quota": 2,
  "used": 2,
  "message": "You have reached your daily limit of 2 jobs. Please try again tomorrow."
}
```

**Status Code:** `429`

## Quick Validation Checklist

- [ ] `.env.example` file exists with `DEFAULT_JOB_QUOTA=50`
- [ ] JobUsage table exists in database
- [ ] JobUsage table has records after generating jobs
- [ ] HTTP 429 returned when quota exceeded
- [ ] Error message includes quota and used count
- [ ] Admin endpoint `/api/admin/usage` accessible (requires admin role)
- [ ] User time online tracking (`last_api_request_at`) updates

## Testing Admin Endpoint

```bash
# First, set a user's role to 'admin' in database:
sqlite3 backend/instance/mesh_app.db "UPDATE users SET role='admin' WHERE email='your@email.com';"

# Then test the endpoint:
curl -H "Authorization: Bearer <admin_jwt_token>" \
  http://localhost:5000/api/admin/usage
```

Expected response:
```json
{
  "period": "2026-01-13 to 2026-01-19",
  "top_users": [
    {
      "user_id": 1,
      "email": "user@example.com",
      "job_count": 15,
      "completed": 14,
      "failed": 1
    }
  ]
}
```

## Notes

- Quota resets at midnight UTC
- All job attempts are logged, even failed ones
- Batch jobs also count toward quota
- Admin endpoint requires `User.role == 'admin'`

