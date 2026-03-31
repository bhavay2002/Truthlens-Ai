from __future__ import annotations

"""
File Name: framenet_extractor_full.py
Description:
    Extracts all useful columns from FrameNet v1.7 XML files including:
        - sentence text
        - frame name
        - lexical unit
        - frame elements
        - span start/end
        - annotation metadata

Output:
    data/interim/framenet_full.csv
"""

from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

FRAMENET_ROOT = Path(
    r"C:\Users\bhava\Downloads\Framenet\framenet_v17\framenet_v17"
)

OUTPUT_FILE = Path("data/interim/framenet_full.csv")


TARGET_DIRS = [
    FRAMENET_ROOT / "fulltext",
    FRAMENET_ROOT / "lu",
    FRAMENET_ROOT / "frame"
]


# ==========================================================
# EXTRACTION FUNCTION
# ==========================================================

def parse_framenet():

    rows = []

    xml_files = []

    for d in TARGET_DIRS:
        xml_files.extend(list(d.rglob("*.xml")))

    print("Total XML files:", len(xml_files))

    for i, file in enumerate(xml_files):

        if i % 200 == 0:
            print(f"Processing {i}/{len(xml_files)}")

        try:

            tree = ET.parse(file)
            root = tree.getroot()

            for sentence in root.iter():

                if not sentence.tag.endswith("sentence"):
                    continue

                sentence_id = sentence.attrib.get("ID")

                sentence_text = None

                for child in sentence:

                    if child.tag.endswith("text"):
                        sentence_text = child.text

                if not sentence_text:
                    continue

                # ======================================
                # annotation sets
                # ======================================

                for ann in sentence.iter():

                    if not ann.tag.endswith("annotationSet"):
                        continue

                    frame_name = ann.attrib.get("frameName")
                    lu_name = ann.attrib.get("luName")
                    ann_id = ann.attrib.get("ID")

                    if frame_name is None:
                        continue

                    # ======================================
                    # layers
                    # ======================================

                    for layer in ann.iter():

                        if not layer.tag.endswith("layer"):
                            continue

                        layer_name = layer.attrib.get("name")

                        for label in layer.iter():

                            if not label.tag.endswith("label"):
                                continue

                            fe_name = label.attrib.get("name")
                            start = label.attrib.get("start")
                            end = label.attrib.get("end")

                            rows.append({

                                "text": sentence_text.strip(),
                                "sentence_id": sentence_id,

                                "frame": frame_name,
                                "lexical_unit": lu_name,

                                "annotation_id": ann_id,
                                "layer": layer_name,

                                "frame_element": fe_name,

                                "span_start": start,
                                "span_end": end,

                                "source_file": file.name

                            })

        except Exception:
            continue

    print("Extracted rows:", len(rows))

    return rows


# ==========================================================
# SAVE DATASET
# ==========================================================

def save_dataset(rows):

    if not rows:
        print("WARNING: No rows extracted.")
        return

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset saved:", OUTPUT_FILE)

    print("\nColumns extracted:\n")

    for c in df.columns:
        print("-", c)

    print("\nPreview:\n")
    print(df.head())


# ==========================================================
# MAIN
# ==========================================================

def main():

    rows = parse_framenet()

    save_dataset(rows)


if __name__ == "__main__":
    main()