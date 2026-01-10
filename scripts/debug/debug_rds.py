import boto3
import json

session = boto3.Session(region_name='us-west-1')
rds = session.client('rds')

try:
    instances = rds.describe_db_instances()
    print("Instances:")
    for i in instances['DBInstances']:
        print(f" - {i['DBInstanceIdentifier']}: {i['DBInstanceStatus']} (VPC: {i['DBSubnetGroup']['VpcId']})")
        
    snapshots = rds.describe_db_snapshots()
    print("\nSnapshots:")
    for s in snapshots['DBSnapshots']:
        if s['DBSnapshotIdentifier'].startswith('manual-promo-'):
            print(f" - {s['DBSnapshotIdentifier']}: {s['Status']}")
except Exception as e:
    print(f"Error: {e}")
