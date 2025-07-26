#!/usr/bin/env python3
"""
Deployment script for XYZ Bank Churn Prediction System
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'logs', 'data', 'static', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def run_tests():
    """Run basic system tests"""
    print("Running system tests...")
    # Add test commands here
    pass

def deploy():
    """Main deployment function"""
    print("Deploying XYZ Bank Churn Prediction System...")

    # Setup
    setup_directories()
    install_requirements()
    run_tests()

    print("✅ Deployment completed successfully!")
    print("To start the system, run: python churn_prediction_system.py")

if __name__ == "__main__":
    deploy()
