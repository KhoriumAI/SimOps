import boto3
import time
import sys

def run_ssm_command(instance_id, commands, region='us-west-1'):
    ssm = boto3.client('ssm', region_name=region)
    print(f"Sending commands to {instance_id}")
    
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': commands}
    )
    
    command_id = response['Command']['CommandId']
    print(f"Command ID: {command_id}")
    
    # Wait for the command to be registered and start
    time.sleep(3)
    
    for _ in range(10):
        try:
            status = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id
            )
            print(f"Status: {status['Status']}")
            if status['Status'] in ['Pending', 'InProgress', 'Delayed']:
                time.sleep(3)
                continue
                
            print("--- STDOUT ---")
            print(status['StandardOutputContent'])
            print("--- STDERR ---")
            print(status['StandardErrorContent'])
            return
        except ssm.exceptions.InvocationDoesNotExist:
            print("Invocation not ready yet...")
            time.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    instance_id = "i-0bdb1e0eaa658cc7c"
    # Just check status and logs
    cmds = [
        'sudo systemctl status mesh-backend',
        'sudo journalctl -u mesh-backend -n 20 --no-pager'
    ]
    run_ssm_command(instance_id, cmds)
