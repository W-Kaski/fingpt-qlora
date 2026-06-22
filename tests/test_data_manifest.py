import json
import tempfile
import unittest
from pathlib import Path


class DataManifestTest(unittest.TestCase):
    def test_manifest_records_split_counts_hashes_and_duplicates(self):
        from src.data.manifest import build_data_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            splits_dir.mkdir()

            train = [
                {
                    "task_type": "sentiment",
                    "conversations": [
                        {"from": "human", "value": "same text"},
                        {"from": "gpt", "value": "Positive"},
                    ],
                },
                {
                    "task_type": "sentiment",
                    "conversations": [
                        {"from": "human", "value": "train only"},
                        {"from": "gpt", "value": "Neutral"},
                    ],
                },
            ]
            val = []
            test = [
                {
                    "task_type": "sentiment",
                    "conversations": [
                        {"from": "human", "value": "same text"},
                        {"from": "gpt", "value": "Positive"},
                    ],
                }
            ]

            for name, data in {"train": train, "val": val, "test": test}.items():
                with open(splits_dir / f"{name}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)

            manifest = build_data_manifest(
                splits_dir=splits_dir,
                dataset_sources={"fingpt_sentiment": "FinGPT/fingpt-sentiment-train"},
                seed=42,
            )

        self.assertEqual(manifest["split_seed"], 42)
        self.assertEqual(manifest["dataset_sources"]["fingpt_sentiment"], "FinGPT/fingpt-sentiment-train")
        self.assertEqual(manifest["splits"]["train"]["num_examples"], 2)
        self.assertEqual(manifest["splits"]["test"]["task_distribution"], {"sentiment": 1})
        self.assertRegex(manifest["splits"]["train"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["duplicate_report"]["train_test_exact_duplicates"], 1)


if __name__ == "__main__":
    unittest.main()
