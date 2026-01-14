import boto3
import json
import time
import sys

STAGING_CONFIG_PATH = r"c:\Users\markm\Downloads\MeshPackageLean\config\aws\staging_cf_config.json"

def main():
    client = boto3.client('cloudfront')
    
    print("\nReading Staging Config template...")
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
    
    print(f"Attempting to create Staging Distribution with aliases: {new_aliases}...")
    
    max_retries = 10
    for i in range(max_retries):
        try:
            response = client.create_distribution(DistributionConfig=staging_config)
            
            new_dist = response['Distribution']
            new_domain = new_dist['DomainName']
            new_id = new_dist['Id']
            
            print(f"✅ Staging Distribution created successfully.")
            print(f"   ID: {new_id}")
            print(f"   Domain: {new_domain}")
            
            # Save this info for the user
            metadata_dir = Path(__file__).parent.parent.parent / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            with open(metadata_dir / 'staging_dist_info.txt', 'w') as f:
                f.write(f"Domain: {new_domain}\nID: {new_id}")
            return

        except client.exceptions.CNAMEAlreadyExists:
            print(f"attempt {i+1}: CNAME still locked. Waiting 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            break

if __name__ == "__main__":
    main()
