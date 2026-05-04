"""
Gemini API wrapper for evaluation.
Parallel to claude_eval.py / OpenAI helpers in evaluate_models.py.
"""
import os
import re
import sys
import time
import random
from google import genai
from google.genai import types

sys.stdout.reconfigure(line_buffering=True)

client = genai.Client()  # reads GEMINI_API_KEY from env

MAX_RETRIES = 3
BASE_BACKOFF = 6.0  # seconds


def _call_with_retry(model, prompt, max_tokens):
    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.0,
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=cfg,
            )
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                backoff = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1.5)
                time.sleep(min(backoff, 120))
                continue
            # Non-retryable
            raise
    # Exhausted retries
    raise RuntimeError(f"retries exhausted: {model}")

GEMINI_MODELS = {
    "gemini-2.0-flash":  "2024-12",
    "gemini-2.5-flash":  "2025-06",
    "gemini-2.5-pro":    "2025-06",
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _clean_choice(c):
    if not isinstance(c, str):
        return str(c)
    return re.sub(r"^\s*[A-Da-d][\)\.\:]\s*", "", c).strip()


def _generate(model, prompt, max_tokens=800):
    return _call_with_retry(model, prompt, max_tokens)


def gemini_eval_mc(model, question, choices, answer):
    """Multiple choice eval via Gemini. Returns 0/1 or None."""
    n = min(len(choices), 26)
    letters = list(LETTERS[:n])
    cleaned = [_clean_choice(c) for c in choices[:n]]
    options_str = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(cleaned))
    prompt = f"{question}\n\n{options_str}\n\nOutput ONLY the letter of the correct answer."
    try:
        text = _generate(model, prompt, max_tokens=10)
        m = re.search(r"[A-Z]", text.upper())
        if m:
            pred_letter = m.group()
            correct_letter = letters[answer] if isinstance(answer, int) else str(answer).upper()
            return 1 if pred_letter == correct_letter else 0
    except Exception:
        return None
    return 0


def gemini_eval_gsm(model, question, correct_final):
    """GSM8K eval via Gemini. Returns 0/1 or None."""
    prompt = (
        "Solve this math word problem step by step. End your answer with '#### <number>' on the final line.\n\n"
        f"{question}"
    )
    try:
        text = _generate(model, prompt, max_tokens=800)
        m = re.search(r"####\s*([-\d,.]+)", text)
        if m:
            pred = m.group(1).strip().replace(",", "")
        else:
            nums = re.findall(r"[-]?\d+\.?\d*", text)
            pred = nums[-1] if nums else None
        if pred is None:
            return 0
        try:
            return 1 if abs(float(pred) - float(str(correct_final).replace(",", ""))) < 1e-4 else 0
        except Exception:
            return 0
    except Exception:
        return None


def gemini_eval_truthfulqa(model, question, choices, answer_idx):
    """TruthfulQA MC eval (same signature as claude's)."""
    return gemini_eval_mc(model, question, choices, answer_idx)


def gemini_eval_humaneval(model, prompt_code):
    """HumanEval code completion. Returns the full completion text."""
    prompt = (
        "Complete the following Python function. Output ONLY a complete Python "
        "function definition (including the original signature and docstring), "
        "no markdown fences, no explanation.\n\n" + prompt_code
    )
    try:
        text = _generate(model, prompt, max_tokens=1000)
        # Strip markdown fences if present
        text = re.sub(r"^```(?:python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text
    except Exception:
        return None
