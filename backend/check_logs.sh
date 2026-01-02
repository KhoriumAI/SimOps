#!/bin/bash
# Check the last 50 lines of gunicorn logs for errors
echo "=== Recent Gunicorn Errors ==="
sudo journalctl -u gunicorn -n 50 --no-pager | grep -A 5 -E "(Error|Traceback|Exception|500)"

echo ""
echo "=== All recent logs (last 30 lines) ==="
sudo journalctl -u gunicorn -n 30 --no-pager
