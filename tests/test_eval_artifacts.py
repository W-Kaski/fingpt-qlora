import json
import tempfile
import unittest
from pathlib import Path


class EvaluationArtifactsTest(unittest.TestCase):
    def test_writes_metrics_predictions_and_comparison_table(self):
        from src.eval.run_evaluation import write_evaluation_artifacts

        results = [
            {
                "task_type": "sentiment",
                "reference": "## Sentiment Analysis\n\n**Sentiment: Positive**",
                "base_prediction": "**Sentiment: Neutral**",
                "ft_prediction": "## Sentiment Analysis\n\n**Sentiment: Positive**",
            },
            {
                "task_type": "sentiment",
                "reference": "## Sentiment Analysis\n\n**Sentiment: Negative**",
                "base_prediction": "**Sentiment: Positive**",
                "ft_prediction": "## Sentiment Analysis\n\n**Sentiment: Negative**",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = write_evaluation_artifacts(
                results=results,
                output_dir=output_dir,
                experiment_name="unit_eval",
                adapter_dataset="unit/adapter",
            )

            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = (output_dir / "predictions.jsonl").read_text().strip().splitlines()
            comparison = (output_dir / "comparison_table.md").read_text()

        self.assertEqual(summary["experiment"], "unit_eval")
        self.assertEqual(metrics["experiment"], "unit_eval")
        self.assertEqual(metrics["adapter_dataset"], "unit/adapter")
        self.assertEqual(metrics["tasks"]["sentiment"]["accuracy"]["fine_tuned"], 1.0)
        self.assertEqual(metrics["tasks"]["sentiment"]["accuracy"]["base"], 0.0)
        self.assertEqual(
            metrics["tasks"]["sentiment"]["labels"],
            ["Negative", "Neutral", "Positive"],
        )
        self.assertEqual(
            metrics["tasks"]["sentiment"]["confusion_matrix"]["fine_tuned"],
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        )
        self.assertIn("Positive", metrics["tasks"]["sentiment"]["per_class"]["fine_tuned"])
        self.assertEqual(
            metrics["tasks"]["sentiment"]["per_class"]["fine_tuned"]["Positive"]["f1"],
            1.0,
        )
        self.assertEqual(len(predictions), 2)
        self.assertIn("Sentiment accuracy", comparison)
        self.assertIn("Macro F1", comparison)
        self.assertIn("Per-class F1", comparison)


if __name__ == "__main__":
    unittest.main()
