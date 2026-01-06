import boto3
import json

session = boto3.Session(region_name='us-east-1')
cf = session.client('cloudfront')

try:
    d = cf.get_distribution(Id='E1K001ACS10874')
    distro = d['Distribution']
    print(f"Distribution ID: {distro['Id']}")
    print(f"Aliases: {distro.get('Aliases', {}).get('Items', [])}")
    
    print("\nOrigins:")
    for o in distro['Origins']['Items']:
        print(f" - ID: {o['Id']}, Domain: {o['DomainName']}")
        
    print("\nBehaviors:")
    default = distro['DefaultCacheBehavior']
    print(f" - Default (*): {default['TargetOriginId']}")
    
    if 'CacheBehaviors' in distro and 'Items' in distro['CacheBehaviors']:
        for b in distro['CacheBehaviors']['Items']:
            print(f" - Path {b['PathPattern']}: {b['TargetOriginId']}")
            
except Exception as e:
    print(f"Error: {e}")
