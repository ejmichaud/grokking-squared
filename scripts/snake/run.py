import subprocess
from sys import argv

# Run
filename = argv[1]
with open(filename, 'r') as f:
    args = dict(f.read())
    args = [item for item in _ for _ in args.items()]

subprocess.run(["python", "scripts/transformer_grokking.py"])
