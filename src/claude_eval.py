"""
Claude API wrapper for evaluation + perturbation generation.
Same interface as OpenAI helpers in evaluate_models.py
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic

sys.stdout.reconfigure(line_buffering=True)
client = Anthropic(timeout=60.0, max_retries=2)

CLAUDE_MODELS = {
    "claude-haiku-4-5":  "2025-09",   # release approximate
    "claude-sonnet-4-5": "2025-09",
    "claude-opus-4-5":   "2025-10",
}


def _clean_choice(c):
    if not isinstance(c, str):
        return str(c)
    return re.sub(r"^\s*[A-Da-d][\)\.\:]\s*", "", c).strip()


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def claude_eval_mc(model, question, choices, answer):
    """Multiple choice eval via Claude. Returns 0/1 or None."""
    n = min(len(choices), 26)
    letters = list(LETTERS[:n])
    cleaned = [_clean_choice(c) for c in choices[:n]]
    options_str = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(cleaned))
    prompt = f"{question}\n\n{options_str}\n\nOutput ONLY the letter of the correct answer."
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        pred = resp.content[0].text.strip().upper()
        m = re.search(r"[A-Z]", pred)
        if m:
            pred_letter = m.group()
            correct_letter = letters[answer] if isinstance(answer, int) else str(answer).upper()
            return 1 if pred_letter == correct_letter else 0
    except Exception:
        return None
    return 0


def claude_eval_gsm(model, question, correct_final):
    """GSM8K eval via Claude. Returns 0/1 or None."""
    prompt = (
        "Solve this math word problem step by step. End your answer with '#### <number>' on the final line.\n\n"
        f"{question}"
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
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
        except:
            return 0
    except Exception:
        return None


def claude_paraphrase_mc(question, choices, answer_idx):
    """Paraphrase a multiple-choice question via Claude. Returns dict or None."""
    correct = LETTERS[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
    choices_str = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    prompt = f"""Paraphrase this multiple-choice question while preserving meaning, difficulty, and the correct answer.

Rules:
1. Paraphrase the question stem with different wording, same meaning.
2. Keep all answer choices semantically identical (light rewording allowed).
3. The correct answer letter must remain {correct}.
4. Do NOT change technical terms, numbers, or named entities.
5. Output ONLY a JSON object with keys "question" (string) and "options" (array of {len(choices)} strings).

Original question:
{question}

Choices:
{choices_str}
"""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Extract JSON (Claude may wrap in markdown)
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return None
        parsed = json.loads(m.group())
        if "question" in parsed and "options" in parsed and len(parsed["options"]) == len(choices):
            return parsed
    except Exception:
        return None
    return None


def claude_paraphrase_gsm(question, answer):
    """Number-substitute a GSM8K problem via Claude. Returns dict or None."""
    prompt = f"""Rewrite this math word problem with substituted numbers and names, preserving the EXACT same solution structure and operation order.

Rules:
1. Replace proper nouns and numbers; keep the operations and step count identical.
2. The new problem should have the same difficulty.
3. Output ONLY a JSON object with keys "question" (string), "answer_text" (worked solution), and "final_answer" (just the number).

Original problem:
{question}

Original solution:
{answer}
"""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return None
        parsed = json.loads(m.group())
        if "question" in parsed and "final_answer" in parsed:
            return {
                "question": parsed["question"],
                "answer": (parsed.get("answer_text", "") + f"\n#### {parsed['final_answer']}"),
                "final_answer": parsed["final_answer"],
            }
    except Exception:
        return None
    return None
