"""
Ensure all dependencies are installed and configured
"""

import os
import subprocess

def ensure_dependencies():
    # check apt dependencies
    subprocess.check_call(['apt-get', 'update'])
    subprocess.check_call(['apt-get', 'install', '-y', 'python3-pip', 'python3-venv'])
    subprocess.check_call(['apt-get', 'install', '-y', 'sqlite3', 'libsqlite3-dev','libfuzzy-dev'])

    # check pip dependencies
    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
