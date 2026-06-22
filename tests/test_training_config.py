import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

try:
    import trl  # noqa: F401
    HAS_TRL = True
except ImportError:
    HAS_TRL = False


class TrainingConfigTest(unittest.TestCase):
    DIAGNOSTIC_YAML = """\
    model:
      name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
      max_seq_length: 2048
      load_in_4bit: true
    lora:
      r: 16
      alpha: 32
      dropout: 0.05
      target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    training:
      experiment_name: "v3_diagnostic"
      batch_size: 4
      gradient_accumulation_steps: 4
      learning_rate: 0.0001
      num_epochs: 1
      max_steps: 10
      scheduler: cosine
      warmup_ratio: 0.1
      weight_decay: 0.01
      optim: adamw_8bit
      fp16: true
      bf16: false
      completion_only_loss: true
      packing: false
      max_train_examples: 128
      max_eval_examples: 64
      save_strategy: "no"
      eval_strategy: "no"
      load_best_model_at_end: false
    evaluation:
      eval_steps: 999999
      save_steps: 999999
      save_total_limit: 1
      logging_steps: 1
    data:
      train_split: 0.85
      val_split: 0.10
      test_split: 0.05
      task_weights:
        sentiment: 1.00
    """

    def test_yaml_diagnostic_fields_map_to_dataclass_config(self):
        from src.config import create_configs_from_yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostic.yaml"
            path.write_text(self.DIAGNOSTIC_YAML, encoding="utf-8")
            model_config, lora_config, data_config, train_config, eval_config = (
                create_configs_from_yaml(str(path))
            )

        self.assertEqual(model_config.base_model, "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
        self.assertEqual(model_config.max_seq_length, 2048)
        self.assertEqual(lora_config.r, 16)
        self.assertEqual(train_config.max_steps, 10)
        self.assertEqual(train_config.save_strategy, "no")
        self.assertEqual(train_config.eval_strategy, "no")
        self.assertFalse(train_config.load_best_model_at_end)
        self.assertTrue(train_config.completion_only_loss)
        self.assertEqual(train_config.max_train_examples, 128)
        self.assertEqual(data_config.train_split, 0.85)

    @unittest.skipUnless(HAS_TRL, "trl not installed")
    def test_yaml_diagnostic_fields_map_to_sft_config(self):
        from src.config import create_configs_from_yaml
        from src.train.train_sft import get_training_args

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostic.yaml"
            path.write_text(self.DIAGNOSTIC_YAML, encoding="utf-8")
            model_config, _, _, train_config, _ = create_configs_from_yaml(str(path))

        args = get_training_args(model_config, train_config, output_suffix="diagnostic")
        self.assertEqual(args.max_steps, 10)
        self.assertEqual(args.save_strategy, "no")
        self.assertEqual(args.eval_strategy, "no")
        self.assertFalse(args.load_best_model_at_end)


if __name__ == "__main__":
    unittest.main()
