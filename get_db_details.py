import boto3

session = boto3.Session(region_name='us-west-1')
rds = session.client('rds')

try:
    instance = rds.describe_db_instances(DBInstanceIdentifier='khorium-webdev-db')['DBInstances'][0]
    print(f"Name: {instance['DBInstanceIdentifier']}")
    print(f"VPC: {instance['DBSubnetGroup']['VpcId']}")
    print(f"Subnet Group: {instance['DBSubnetGroup']['DBSubnetGroupName']}")
    print(f"SGs: {[sg['VpcSecurityGroupId'] for sg in instance['VpcSecurityGroups']]}")
except Exception as e:
    print(f"Error: {e}")
