SYSTEM_PROMPTS = {
    "sentiment": (
        "You are a financial sentiment analyst. Classify the sentiment of "
        "financial text as Positive, Negative, or Neutral. Explain your "
        "reasoning concisely."
    ),
    "summary": (
        "You are a financial report analyst. Summarize earnings reports and "
        "financial filings into clear, structured briefs with key metrics "
        "highlighted."
    ),
    "qa": (
        "You are a knowledgeable financial advisor. Answer financial questions "
        "accurately with relevant context and caveats."
    ),
    "reasoning": (
        "You are a financial data analyst. Perform step-by-step numerical "
        "reasoning on financial data. Show your calculation process clearly."
    ),
    "analysis": (
        "You are an investment research analyst. Provide balanced analysis "
        "covering both bull and bear cases, key risks, and relevant metrics."
    ),
}

TASK_DESCRIPTIONS = {
    "sentiment": "Analyze the sentiment of this financial text:",
    "summary": "Summarize the following financial report or filing:",
    "qa": None,  # Questions come directly from the dataset
    "reasoning": "Perform step-by-step numerical reasoning on the following financial data:",
    "analysis": "Provide a comprehensive investment analysis for the following:",
}


def get_system_prompt(task_type: str) -> str:
    if task_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown task type: {task_type}. Must be one of {list(SYSTEM_PROMPTS)}")
    return SYSTEM_PROMPTS[task_type]


def get_task_description(task_type: str) -> str:
    if task_type not in TASK_DESCRIPTIONS:
        raise ValueError(f"Unknown task type: {task_type}. Must be one of {list(TASK_DESCRIPTIONS)}")
    return TASK_DESCRIPTIONS[task_type]
