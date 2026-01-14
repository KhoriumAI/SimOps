import boto3
import time
import sys

def run_ssm_command(instance_id, commands, region='us-west-1'):
    ssm = boto3.client('ssm', region_name=region)
    print(f"Sending commands to {instance_id}: {commands}")
    
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': commands}
    )
    
    command_id = response['Command']['CommandId']
    print(f"Command ID: {command_id}")
    
    while True:
        status = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id
        )
        if status['Status'] in ['Pending', 'InProgress', 'Delayed']:
            print(f"Status: {status['Status']}")
            time.sleep(2)
            continue
            
        print(f"Final Status: {status['Status']}")
        print("--- STDOUT ---")
        print(status['StandardOutputContent'])
        print("--- STDERR ---")
        print(status['StandardErrorContent'])
        break

if __name__ == "__main__":
    instance_id = "i-0bdb1e0eaa658cc7c"
    cmds = [
        'cd /home/ec2-user/backend',
        'sudo -u ec2-user git pull origin main',
        'sudo systemctl restart mesh-backend',
        'sudo systemctl status mesh-backend'
    ]
    run_ssm_command(instance_id, cmds)
