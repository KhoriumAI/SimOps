# AWS Deployment Guide for Khorium MeshGen

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CloudFront │────▶│     S3      │     │    RDS      │
│    (CDN)    │     │  (Frontend) │     │ (PostgreSQL)│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│   Route53   │────▶│     ALB     │────▶┌──────┴──────┐
│   (DNS)     │     │(Load Balancer)│   │     EC2     │
└─────────────┘     └─────────────┘    │  (Backend)  │
                                        └──────┬──────┘
                                               │
                                        ┌──────┴──────┐
                                        │     S3      │
                                        │(File Storage)│
                                        └─────────────┘
```

## AWS Services Required

| Service | Purpose | Estimated Cost |
|---------|---------|----------------|
| **EC2** | Backend API server | $15-50/mo (t3.medium) |
| **RDS PostgreSQL** | Database | $15-30/mo (db.t3.micro) |
| **S3** | File storage (CAD files, meshes) | $5-20/mo |
| **CloudFront** | CDN for frontend | $5-10/mo |
| **ALB** | Load balancer | $16/mo + data |
| **Route53** | DNS | $0.50/mo per zone |
| **ACM** | SSL Certificates | Free |

**Estimated Total: $50-130/month** (depending on usage)

---

## Step 1: Set Up AWS RDS (PostgreSQL)

### 1.1 Create RDS Instance

```bash
# Using AWS CLI
aws rds create-db-instance \
    --db-instance-identifier khorium-meshgen-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15.4 \
    --master-username meshgen_admin \
    --master-user-password YOUR_SECURE_PASSWORD \
    --allocated-storage 20 \
    --storage-type gp3 \
    --vpc-security-group-ids sg-xxxxxxxx \
    --db-subnet-group-name your-subnet-group \
    --backup-retention-period 7 \
    --no-publicly-accessible
```

### 1.2 Or via AWS Console:
1. Go to **RDS** → **Create database**
2. Choose **PostgreSQL**
3. Select **Free tier** or **Production** template
4. Settings:
   - DB instance identifier: `khorium-meshgen-db`
   - Master username: `meshgen_admin`
   - Master password: (generate a strong password)
5. Instance: `db.t3.micro` (or larger for production)
6. Storage: 20GB GP3 SSD
7. Connectivity:
   - VPC: Your VPC
   - Subnet group: Create or select
   - Public access: **No** (EC2 will access privately)
8. Create database

### 1.3 Create Database and User

Connect via EC2 or bastion host:
```bash
psql -h your-rds-endpoint.region.rds.amazonaws.com -U meshgen_admin -d postgres

# In psql:
CREATE DATABASE khorium_meshgen;
\c khorium_meshgen

# Create tables (run your Flask migration)
```

---

## Step 2: Set Up S3 Bucket

### 2.1 Create S3 Bucket

```bash
aws s3 mb s3://khorium-meshgen-files --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
    --bucket khorium-meshgen-files \
    --versioning-configuration Status=Enabled

# Set lifecycle policy to clean old files
aws s3api put-bucket-lifecycle-configuration \
    --bucket khorium-meshgen-files \
    --lifecycle-configuration file://lifecycle.json
```

### 2.2 S3 Bucket Policy

Create `bucket-policy.json`:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowEC2Access",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::YOUR_ACCOUNT_ID:role/EC2MeshGenRole"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::khorium-meshgen-files",
                "arn:aws:s3:::khorium-meshgen-files/*"
            ]
        }
    ]
}
```

### 2.3 CORS Configuration for S3

```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
        "AllowedOrigins": ["https://your-domain.com"],
        "ExposeHeaders": ["ETag"]
    }
]
```

---

## Step 3: Set Up EC2 Instance

### 3.1 Launch EC2 Instance

1. **AMI**: Amazon Linux 2023 or Ubuntu 22.04
2. **Instance Type**: `t3.medium` (2 vCPU, 4GB RAM) - for mesh generation
3. **Storage**: 30GB GP3 SSD
4. **Security Group**:
   - Inbound: SSH (22), HTTP (80), HTTPS (443) from ALB
   - Outbound: All traffic
5. **IAM Role**: Create role with S3 and RDS access

### 3.2 Install Dependencies on EC2

```bash
# Update system
sudo yum update -y  # Amazon Linux
# OR
sudo apt update && sudo apt upgrade -y  # Ubuntu

# Install Python 3.11+
sudo yum install python3.11 python3.11-pip -y
# OR
sudo apt install python3.11 python3.11-venv python3-pip -y

# Install system dependencies for gmsh
sudo yum install -y mesa-libGL libXrender libXcursor libXinerama libXi
# OR
sudo apt install -y libgl1-mesa-glx libxrender1 libxcursor1 libxinerama1 libxi6

# Clone your repository
git clone https://github.com/YOUR_USERNAME/MeshPackageLean.git
cd MeshPackageLean

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn boto3 psycopg2-binary
```

### 3.3 Create Production Environment File

Create `/home/ec2-user/MeshPackageLean/backend/.env`:

```bash
# Environment
FLASK_ENV=production

# Database - AWS RDS
DB_HOST=your-rds-endpoint.region.rds.amazonaws.com
DB_PORT=5432
DB_NAME=khorium_meshgen
DB_USER=meshgen_admin
DB_PASSWORD=your-secure-password

# AWS S3
USE_S3=true
AWS_REGION=us-west-1
S3_BUCKET_NAME=muaz-webdev-assets
# Files stored as: {user_email}/uploads/ and {user_email}/mesh/
# Note: Use IAM role instead of access keys on EC2

# Security (generate with: python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=your-generated-secret-key
JWT_SECRET_KEY=your-generated-jwt-key

# CORS
CORS_ORIGINS=https://your-domain.com
```

### 3.4 Run with Gunicorn

Create `/etc/systemd/system/meshgen.service`:

```ini
[Unit]
Description=Khorium MeshGen API
After=network.target

[Service]
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/MeshPackageLean/backend
Environment="PATH=/home/ec2-user/MeshPackageLean/venv/bin"
ExecStart=/home/ec2-user/MeshPackageLean/venv/bin/gunicorn \
    --workers 4 \
    --bind 0.0.0.0:5000 \
    --timeout 300 \
    "api_server:create_app()"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable meshgen
sudo systemctl start meshgen

# Check status
sudo systemctl status meshgen
```

---

## Step 4: Set Up Application Load Balancer (ALB)

1. Go to **EC2** → **Load Balancers** → **Create Load Balancer**
2. Choose **Application Load Balancer**
3. Configure:
   - Name: `meshgen-alb`
   - Scheme: Internet-facing
   - Listeners: HTTP (80), HTTPS (443)
4. Add Target Group:
   - Target type: Instances
   - Protocol: HTTP, Port: 5000
   - Health check: `/api/health`
5. Register your EC2 instance

---

## Step 5: Deploy Frontend to S3 + CloudFront

### 5.1 Build Frontend

```bash
cd web-frontend

# Copy and edit the environment file
cp env.production.example .env.production

# Edit .env.production with your API domain:
# VITE_API_URL=https://api.your-domain.com

# Or set environment variable directly:
# VITE_API_URL=https://api.your-domain.com npm run build

# Install dependencies
npm install

# Build for production
npm run build
```

### 5.2 Upload to S3

```bash
# Create frontend bucket
aws s3 mb s3://khorium-meshgen-frontend

# Upload build
aws s3 sync dist/ s3://khorium-meshgen-frontend --delete

# Enable static website hosting
aws s3 website s3://khorium-meshgen-frontend \
    --index-document index.html \
    --error-document index.html
```

### 5.3 Create CloudFront Distribution

1. Go to **CloudFront** → **Create Distribution**
2. Origin: S3 bucket (use S3 website endpoint)
3. Settings:
   - Price class: Use only North America and Europe (cheaper)
   - Alternate domain: `your-domain.com`
   - SSL Certificate: Request from ACM
4. Default behavior:
   - Viewer protocol policy: Redirect HTTP to HTTPS
   - Cache policy: CachingOptimized

---

## Step 6: Set Up Route53 (DNS)

1. Create Hosted Zone for your domain
2. Add records:
   - `your-domain.com` → CloudFront distribution (A record, Alias)
   - `api.your-domain.com` → ALB (A record, Alias)

---

## Step 7: Database Migration

```bash
# SSH into EC2
ssh ec2-user@your-ec2-ip

cd MeshPackageLean
source venv/bin/activate

# Run migrations
cd backend
flask db upgrade

# Or initialize fresh database
python -c "from api_server import create_app; from models import db; app = create_app(); app.app_context().push(); db.create_all()"
```

---

## Environment Variables Summary

```bash
# Required for Production
FLASK_ENV=production
DB_HOST=xxx.region.rds.amazonaws.com
DB_PORT=5432
DB_NAME=khorium_meshgen
DB_USER=meshgen_admin
DB_PASSWORD=secure-password
USE_S3=true
AWS_REGION=us-west-1
S3_BUCKET_NAME=muaz-webdev-assets
SECRET_KEY=generated-secret
JWT_SECRET_KEY=generated-jwt-secret
CORS_ORIGINS=https://your-domain.com
```

---

## Monitoring & Logs

### CloudWatch Logs

```bash
# Install CloudWatch agent on EC2
sudo yum install amazon-cloudwatch-agent -y

# Configure to stream application logs
```

### Health Checks

- ALB health check: `/api/health`
- RDS: Enable Enhanced Monitoring
- S3: Enable access logging

---

## Cost Optimization Tips

1. **EC2**: Use Reserved Instances for 30-60% savings
2. **RDS**: Use Reserved Instances
3. **S3**: Set lifecycle policies to move old files to Glacier
4. **CloudFront**: Use appropriate price class
5. **NAT Gateway**: Consider NAT Instance for lower traffic

---

## Security Checklist

- [ ] RDS not publicly accessible
- [ ] Security groups restrict access
- [ ] IAM roles instead of access keys
- [ ] SSL/TLS everywhere
- [ ] Secrets in AWS Secrets Manager
- [ ] Enable CloudTrail logging
- [ ] Regular security updates
- [ ] Database backups enabled
