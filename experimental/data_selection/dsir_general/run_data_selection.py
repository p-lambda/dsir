import subprocess
import os

environment = dict(os.environ, PYTHONHASHSEED='42')
subprocess.run(["python", "data_selection.py"], env=environment)
