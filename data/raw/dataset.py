import json
import pandas as pd
from pathlib import Path

rows = []

for file in Path("FakeNewsNet").rglob("news content.json"):
    
    with open(file) as f:
        data = json.load(f)

    rows.append({
        "title": data.get("title"),
        "text": data.get("text"),
        "label": 1 if "fake" in str(file) else 0
    })

df = pd.DataFrame(rows)

df.to_csv("fakenewsnet.csv", index=False)