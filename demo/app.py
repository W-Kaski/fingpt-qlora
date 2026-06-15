"""FinGPT Gradio Demo - Financial Analysis Chatbot."""

import os
import gradio as gr
from unsloth import FastLanguageModel

MODEL_PATH = os.environ.get("MODEL_PATH", "W-Kaski/fingpt-qlora-qwen2.5-7b")

SYSTEM_PROMPT = (
    "You are FinGPT, a financial analysis assistant. You provide clear, "
    "structured analysis of financial data, reports, and market information. "
    "You can perform sentiment analysis, summarize financial reports, answer "
    "financial questions, and provide investment analysis."
)

EXAMPLES = [
    "Analyze the sentiment: Tesla shares surged 12% after record Q3 deliveries",
    "What are the key risks of investing in emerging market bonds?",
    "Summarize Apple's Q4 2024 earnings: Revenue $89.5B (+8% YoY), EPS $1.64 (beat by $0.08)",
    "Explain dollar-cost averaging and when investors should use it",
    "Compare the risk-return profile of growth stocks vs value stocks",
]

# Global model references (loaded once at startup)
_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    print(f"Loading model from {MODEL_PATH}...")
    try:
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from {MODEL_PATH}: {e}")
        raise


def respond(message, history):
    if _model is None or _tokenizer is None:
        return "Error: Model failed to load. Check the logs for details."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    output_ids = _model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = _tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    return response


demo = gr.ChatInterface(
    fn=respond,
    title="FinGPT - Financial Analysis Chatbot",
    description=(
        "Fine-tuned Qwen2.5-7B with QLoRA for financial analysis. "
        "Supports sentiment analysis, report summarization, financial Q&A, "
        "and investment analysis."
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 800px;
        margin: auto;
    }
    """,
)

if __name__ == "__main__":
    load_model()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
