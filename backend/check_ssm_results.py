import boto3
import time

def check_and_get_result(instance_id, region='us-west-1'):
    ssm = boto3.client('ssm', region_name=region)
    
    # Get all recent commands for this instance
    response = ssm.list_commands(
        InstanceId=instance_id,
        MaxResults=5
    )
    
    for cmd in response['Commands']:
        cmd_id = cmd['CommandId']
        status = cmd['Status']
        print(f"Command {cmd_id}: {status}")
        
        if status in ['Success', 'Failed']:
            inv = ssm.get_command_invocation(
                CommandId=cmd_id,
                InstanceId=instance_id
            )
            print(f"\n--- Output for {cmd_id} ---")
            print(inv.get('StandardOutputContent', '(no stdout)'))
            if inv.get('StandardErrorContent'):
                print("--- STDERR ---")
                print(inv['StandardErrorContent'])
            print("-" * 40)

if __name__ == "__main__":
    check_and_get_result("i-0bdb1e0eaa658cc7c")
