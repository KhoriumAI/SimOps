import boto3
import json

session = boto3.Session(region_name='us-west-1')
elbv2 = session.client('elbv2')

alb_arn = 'arn:aws:elasticloadbalancing:us-west-1:571832839665:loadbalancer/app/webdev-alb-stg/f198da4d96a3d417'

try:
    listeners = elbv2.describe_listeners(LoadBalancerArn=alb_arn)
    print(json.dumps(listeners['Listeners'], indent=2, default=str))
except Exception as e:
    print(f"Error: {e}")
