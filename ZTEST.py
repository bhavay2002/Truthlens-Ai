from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
import pandas as pd


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

FRAMENET_ROOT = Path(
    r"C:\Users\bhava\Downloads\Framenet\framenet_v17\framenet_v17"
)

OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / ".csv"


# -------------------------------------------------------
# Parse XML file
# -------------------------------------------------------

def parse_document(xml_file: Path):

    rows: List[dict] = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception:
        return rows

    ns = {"fn": "http://framenet.icsi.berkeley.edu"}

    for sentence in root.findall(".//fn:sentence", ns):

        text_node = sentence.find("fn:text", ns)

        if text_node is None:
            continue

        text = text_node.text

        for ann in sentence.findall(".//fn:annotationSet", ns):

            frame_name = ann.attrib.get("frameName")

            if frame_name is None:
                continue

            actor = None
            target = None
            trigger = None

            for layer in ann.findall("fn:layer", ns):

                if layer.attrib.get("name") == "Target":

                    for label in layer.findall("fn:label", ns):
                        trigger = label.attrib.get("name")

                if layer.attrib.get("name") == "FE":

                    for label in layer.findall("fn:label", ns):

                        role = label.attrib.get("name")

                        if role in ["Agent", "Actor", "Attacker"]:
                            actor = role

                        if role in ["Patient", "Victim", "Target"]:
                            target = role

            rows.append(
                {
                    "text": text,
                    "frame": frame_name,
                    "trigger": trigger,
                    "actor_role": actor,
                    "target_role": target,
                }
            )

    return rows


# -------------------------------------------------------
# Build dataset
# -------------------------------------------------------

def build_dataset():

    rows: List[dict] = []

    xml_files = list(FRAMENET_ROOT.rglob("*.xml"))

    print("Total XML files found:", len(xml_files))

    for xml_file in xml_files:

        rows.extend(parse_document(xml_file))

    df = pd.DataFrame(rows)

    df = df.drop_duplicates()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Total extracted rows:", len(df))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":

    build_dataset()