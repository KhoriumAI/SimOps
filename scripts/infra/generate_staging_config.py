from pathlib import Path
CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "aws"

with open(CONFIG_DIR / 'dev_cf_config.json', 'r') as f:
    dev_data = json.load(f)

config = dev_data['DistributionConfig']

# 1. Update Aliases
config['Aliases']['Items'] = ['staging.khorium.ai']

# 2. Update Origins
backend_origin = config['Origins']['Items'][0]
backend_origin['Id'] = 'ALB-Staging-Backend'
backend_origin['DomainName'] = 'webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com'
backend_origin['CustomOriginConfig']['HTTPPort'] = 80 # Staging ALB is likely on 80

s3_origin = config['Origins']['Items'][1]
s3_origin_id = 'S3-Staging-Frontend'
s3_origin['Id'] = s3_origin_id
s3_origin['DomainName'] = 'muaz-mesh-web-staging.s3-website-us-west-1.amazonaws.com'

# 3. Update Default Cache Behavior
config['DefaultCacheBehavior']['TargetOriginId'] = s3_origin_id

# 4. Update Cache Behaviors
for behavior in config['CacheBehaviors']['Items']:
    behavior['TargetOriginId'] = 'ALB-Staging-Backend'

# 5. Set unique CallerReference
config['CallerReference'] = f"staging-promo-{int(time.time())}"

# 6. Remove ETag (not needed for creation) và other fields if necessary
# config.pop('WebACLId', None) # Maybe keep it?

with open(CONFIG_DIR / 'staging_cf_config.json', 'w') as f:
    json.dump(config, f, indent=4)
