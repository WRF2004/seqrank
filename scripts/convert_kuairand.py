"""Stream official KuaiRand logs into the impression schema (no fabricated conversion).

Example: python scripts/convert_kuairand.py --input log_standard.csv --output data/events.csv
Keep random-exposure logs separate; do not mix traffic policies silently.
"""

import argparse
import csv
from pathlib import Path


def convert(sources, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["user_id", "item_id", "timestamp", "click", "conversion", "dwell"])
        for path in sources:
            with open(path, newline="") as source:
                rows = csv.DictReader(source)
                required = {"user_id", "video_id", "time_ms", "is_click", "play_time_ms"}
                if not required <= set(rows.fieldnames or []):
                    raise ValueError(f"{path}: missing {required - set(rows.fieldnames or [])}")
                for row in rows:
                    writer.writerow(
                        [
                            row["user_id"],
                            row["video_id"],
                            row["time_ms"],
                            row["is_click"],
                            -1,
                            float(row["play_time_ms"]) / 1000,
                        ]
                    )
                    count += 1
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print({"rows": convert(args.input, args.output), "conversion_labels": "unavailable"})
