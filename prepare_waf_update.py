
import json

try:
    with open('web_acl.json', 'r', encoding='utf-8-sig') as f: # Handle potential BOM from PowerShell output
        data = json.load(f)
except json.JSONDecodeError:
    with open('web_acl.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

rules = data['WebACL']['Rules']
lock_token = data['LockToken']

with open('waf_rules_only.json', 'w', encoding='utf-8') as f:
    json.dump(rules, f)

print(f"TOKEN:{lock_token}")
