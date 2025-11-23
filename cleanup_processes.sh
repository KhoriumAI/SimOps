#!/bin/bash
# Cleanup script for orphaned mesh generation processes

echo "Checking for orphaned Python/gmsh processes..."

# Find Python processes (exclude VS Code and this script)
python_pids=$(ps aux | grep -E "[p]ython.*mesh|[p]ython.*gui_final" | grep -v "Code Helper" | awk '{print $2}')

if [ -z "$python_pids" ]; then
    echo "✓ No orphaned Python mesh processes"
else
    echo "Found orphaned Python processes:"
    ps aux | grep -E "[p]ython.*mesh|[p]ython.*gui_final" | grep -v "Code Helper"
    echo ""
    read -p "Kill these processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$python_pids" | xargs kill -9 2>/dev/null
        echo "✓ Processes killed"
    fi
fi

# Check for gmsh processes
gmsh_pids=$(ps aux | grep "[g]msh" | awk '{print $2}')

if [ -z "$gmsh_pids" ]; then
    echo "✓ No orphaned gmsh processes"
else
    echo "Found orphaned gmsh processes:"
    ps aux | grep "[g]msh"
    echo ""
    echo "$gmsh_pids" | xargs kill -9 2>/dev/null
    echo "✓ gmsh processes killed"
fi

echo ""
echo "Cleanup complete!"
