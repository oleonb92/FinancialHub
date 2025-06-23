#!/usr/bin/env python3
"""
Script to automatically download spaCy models after installation.
Run this after: pip install -r requirements.txt
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Installing spaCy language models...")
    
    # List of models to install
    models = [
        "en_core_web_sm",  # English small model
        # "es_core_news_sm",  # Spanish small model (uncomment if needed)
        # "fr_core_news_sm",  # French small model (uncomment if needed)
    ]
    
    success_count = 0
    
    for model in models:
        print(f"\n📦 Installing {model}...")
        if run_command(f"python -m spacy download {model}"):
            success_count += 1
    
    print(f"\n🎉 Installation complete! {success_count}/{len(models)} models installed successfully.")
    
    if success_count == len(models):
        print("✅ All spaCy models installed successfully!")
    else:
        print("⚠️  Some models failed to install. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 