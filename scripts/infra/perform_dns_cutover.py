import boto3
import json
import time
import sys
from datetime import datetime

DEV_DIST_ID = "E352AHA7L040MU"
STAGING_CONFIG_PATH = r"c:\Users\markm\Downloads\MeshPackageLean\config\aws\staging_cf_config.json"

def main():
    client = boto3.client('cloudfront')

    # 1. Update Dev Distribution
    print(f"Fetching configuration for Dev Distribution: {DEV_DIST_ID}...")
    try:
        dev_dist = client.get_distribution_config(Id=DEV_DIST_ID)
        etag = dev_dist['ETag']
        config = dev_dist['DistributionConfig']
        
        # Modify Aliases
        print("Current Dev Aliases:", config.get('Aliases', {}).get('Items', []))
        
        # We want ONLY development.khorium.ai
        config['Aliases'] = {
            'Quantity': 1,
            'Items': ['development.khorium.ai']
        }
        
        print("Updating Dev Distribution to use alias: development.khorium.ai...")
        # Remove app.khorium.ai to free it up
        client.update_distribution(
            Id=DEV_DIST_ID,
            DistributionConfig=config,
            IfMatch=etag
        )
        print("✅ Dev Distribution updated successfully.")
        
    except Exception as e:
        print(f"❌ Failed to update Dev Distribution: {e}")
        # If we fail here, we should probably stop, or manual intervention is needed
        # But if the failure is "alias already exists" (unlikely) or something, we need to know.
        # If app.khorium.ai is NOT removed, we can't add it to Staging.
        return

    # 2. Create Staging Distribution
    print("\nReading Staging Config template...")
    try:
        with open(STAGING_CONFIG_PATH, 'r') as f:
            staging_config = json.load(f)
            
        # Update Aliases
        new_aliases = ['meshgen.khorium.ai', 'app.khorium.ai', 'staging.khorium.ai']
        staging_config['Aliases'] = {
            'Quantity': len(new_aliases),
            'Items': new_aliases
        }
        
        # Ensure unique CallerReference
        staging_config['CallerReference'] = f"staging-dist-{int(time.time())}"
        
        print(f"Creating Staging Distribution with aliases: {new_aliases}...")
        response = client.create_distribution(DistributionConfig=staging_config)
        
        new_dist = response['Distribution']
        new_domain = new_dist['DomainName']
        new_id = new_dist['Id']
        
        print(f"✅ Staging Distribution created successfully.")
        print(f"   ID: {new_id}")
        print(f"   Domain: {new_domain}")
        
        # Save this info for the user
        from pathlib import Path
        metadata_dir = Path(__file__).parent.parent.parent / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        with open(metadata_dir / 'staging_dist_info.txt', 'w') as f:
            f.write(f"Domain: {new_domain}\nID: {new_id}")

    except Exception as e:
        print(f"❌ Failed to create Staging Distribution: {e}")

if __name__ == "__main__":
    main()
