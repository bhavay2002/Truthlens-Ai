from pathlib import Path

folder = Path(r"C:\Users\bhava\OneDrive\Desktop\demo.cpp\Truthlens Ai\data\splits\propaganda2")

print("Files in folder:\n")

for f in folder.iterdir():
    print(f.name)