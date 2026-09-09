import numpy as np
from pathlib import Path
import json

ignore_p1 = Path()
ignore_p2 = Path()
ignore_pcomb = Path()


with open(ignore_p1, "r", encoding="utf-8") as f:
    f1 = json.load(f)

with open(ignore_p2, "r", encoding="utf-8") as f:
    f2 = json.load(f)

f_combined = f1 + f2 #possible because they are lists

with open(ignore_pcomb, "w", encoding="utf-8") as f:
    json.dump(f_combined, f, indent=4, ensure_ascii=False)

print(f"Combined: {len(f1)} + {len(f2)} = {len(f_combined)} json entries.")

