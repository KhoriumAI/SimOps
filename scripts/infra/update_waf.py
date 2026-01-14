
import subprocess
import json
import os

# Ensure PAGER is off
os.environ["AWS_PAGER"] = ""

cmd = [
    "aws", "wafv2", "update-web-acl",
    "--name", "CreatedByCloudFront-f91eec57",
    "--scope", "CLOUDFRONT",
    "--id", "bb4fafa4-0dc7-464d-9d8e-c8a7564ecdd6",
    "--default-action", "Allow={}",
    "--lock-token", "1a826d52-1fdd-40b2-a831-eb0fa765990c",
    "--rules", "file://waf_rules_only.json",
    "--region", "us-east-1",
    "--visibility-config", "SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=CreatedByCloudFront-f91eec57"
]

print("Running WAF update...")
result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

if result.returncode != 0:
    print("Command failed!")
    exit(1)
else:
    print("Success!")
