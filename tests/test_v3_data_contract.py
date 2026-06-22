import subprocess
import sys
import unittest


class PromptCompletionContractTest(unittest.TestCase):
    def test_sharegpt_example_converts_to_prompt_completion(self):
        from src.data.prompt_completion import sharegpt_to_prompt_completion

        example = {
            "conversations": [
                {"from": "system", "value": "System prompt"},
                {"from": "human", "value": "Analyze this"},
                {"from": "gpt", "value": "## Sentiment Analysis"},
            ],
            "task_type": "sentiment",
        }

        converted = sharegpt_to_prompt_completion(example)

        self.assertEqual(
            converted,
            {
                "prompt": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Analyze this"},
                ],
                "completion": [
                    {"role": "assistant", "content": "## Sentiment Analysis"},
                ],
                "task_type": "sentiment",
            },
        )

    def test_sentiment_response_is_stable_across_python_processes(self):
        code = (
            "from src.data.format_chat import format_sentiment_response;"
            "print(format_sentiment_response('Positive', 'Revenue rose 12% to $10B.'))"
        )

        first = subprocess.check_output([sys.executable, "-c", code], text=True)
        second = subprocess.check_output([sys.executable, "-c", code], text=True)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
