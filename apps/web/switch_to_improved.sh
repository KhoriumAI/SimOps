#!/bin/bash
# Switch to improved web app version

echo "🔧 Switching to improved web app..."

# Backup old version
if [ -f "app.py" ]; then
    echo "📦 Backing up old app.py → app_old.py"
    mv app.py app_old.py
fi

# Use improved version
if [ -f "app_improved.py" ]; then
    echo "✅ Activating app_improved.py → app.py"
    cp app_improved.py app.py
    echo ""
    echo "✨ Done! You can now run:"
    echo "   streamlit run app.py"
    echo ""
    echo "Or test the improved version directly:"
    echo "   streamlit run app_improved.py"
else
    echo "❌ Error: app_improved.py not found!"
    exit 1
fi
