import boto3
import json

session = boto3.Session(region_name='us-east-1') # CloudFront is global but usually managed here
cf = session.client('cloudfront')

try:
    distros = cf.list_distributions()
    print("Found distributions:")
    for d in distros.get('DistributionList', {}).get('Items', []):
        aliases = d.get('Aliases', {}).get('Items', [])
        print(f" - ID: {d['Id']}, Domain: {d['DomainName']}, Aliases: {aliases}")
        
        # Check origins
        origins = [o['DomainName'] for o in d['Origins']['Items']]
        print(f"   Origins: {origins}")
except Exception as e:
    print(f"Error: {e}")
