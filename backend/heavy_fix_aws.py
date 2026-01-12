import boto3
import time

def run_fix_and_restart(instance_id, region='us-west-1'):
    ssm = boto3.client('ssm', region_name=region)
    print(f"Repairing instance {instance_id}...")
    
    # 1. Kill any stuck processes, pull clean code, restart service
    cmds = [
        'cd /home/ec2-user/backend',
        'sudo pkill -f mesh_worker_subprocess',  # Kill any stuck workers
        'sudo git checkout .',                  # Discard local changes
        'sudo git pull origin main',             # Get the latest fixes
        'sudo systemctl restart mesh-backend'    # Fresh restart
    ]
    
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': cmds}
    )
    
    cmd_id = response['Command']['CommandId']
    print(f"Fix request sent. Command ID: {cmd_id}")
    
    # Wait for completion
    for i in range(15):
        time.sleep(4)
        status = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
        print(f"Status: {status['Status']}")
        if status['Status'] in ['Success', 'Failed', 'Cancelled']:
            if status['Status'] == 'Success':
                print("--- SUCCESS ---")
                print(status['StandardOutputContent'])
            else:
                print("--- ERROR ---")
                print(status['StandardErrorContent'])
            return True
            
    print("Command is still running or pending. Backend might be slow but fix is in progress.")
    return False

if __name__ == "__main__":
    run_fix_and_restart("i-0bdb1e0eaa658cc7c")
