import os

for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.pt') and not 'venv' in root:
            print(os.path.join(root, f))
