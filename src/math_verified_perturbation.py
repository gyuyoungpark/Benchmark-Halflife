"""
MATH verified perturbation: Generate number-substituted versions with
VERIFIED solutions. The key issue with current perturbations is that
number substitution can make problems unsolvable or change difficulty.

This script:
1. Takes each MATH item
2. Asks Claude to substitute numbers AND solve the new problem
3. Verifies the answer is correct by asking a second model to solve independently
4. Only keeps perturbations where both solutions agree

Goal: Raise fidelity from r≈0.3 to r>0.7
"""
import json, os, sys, re, time
import numpy as np
from pathlib import Path
from anthropic import Anthropic

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_FILE = PERT_DIR / "math_verified_perturbed.json"

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    timeout=60.0, max_retries=2
)


def generate_verified_perturbation(question, answer, topic):
    """Generate a number-substituted version with verified solution."""
    prompt = f"""You are a math teacher. Given a math problem, create a NUMBER-SUBSTITUTED version:

RULES:
1. Replace ALL numbers in the problem with DIFFERENT numbers
2. Keep the EXACT same mathematical structure and operations
3. The new problem MUST be solvable with the same method
4. Solve the new problem step by step
5. The difficulty must be EQUIVALENT (not harder, not easier)

ORIGINAL PROBLEM:
{question}

ORIGINAL ANSWER: {answer}

Output a JSON object with EXACTLY these fields:
{{"question": "the new problem with substituted numbers", "solution": "step-by-step solution of the NEW problem", "final_answer": "just the numerical/algebraic answer"}}

Make sure your solution is CORRECT."""

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=1500,
            messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text.strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return None
        parsed = json.loads(m.group())
        if "question" in parsed and "final_answer" in parsed:
            return parsed
    except Exception as e:
        print(f"    GEN ERR: {str(e)[:80]}", flush=True)
    return None


def verify_perturbation(question, expected_answer):
    """Ask a different model to independently solve the perturbed problem."""
    prompt = f"""Solve this math problem. Show your work, then give ONLY the final answer after "ANSWER:".

{question}"""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=1000,
            messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text.strip()

        # Extract answer after "ANSWER:"
        if "ANSWER:" in text.upper():
            verify_ans = text.upper().split("ANSWER:")[-1].strip()
        else:
            # Try last line
            verify_ans = text.strip().split("\n")[-1].strip()

        # Normalize both answers for comparison
        def normalize(s):
            s = str(s).strip().lower()
            s = re.sub(r'[\\$,\s]', '', s)
            s = s.replace('\\frac', '').replace('\\', '')
            try:
                return float(s)
            except:
                return s

        expected_norm = normalize(expected_answer)
        verify_norm = normalize(verify_ans)

        if expected_norm == verify_norm:
            return True, verify_ans
        # Try numeric comparison
        try:
            if abs(float(expected_norm) - float(verify_norm)) < 0.01:
                return True, verify_ans
        except:
            pass
        return False, verify_ans
    except Exception as e:
        print(f"    VERIFY ERR: {str(e)[:80]}", flush=True)
        return False, None


def main():
    items = json.load(open(DATA_DIR / "items" / "math_lvl5_items.json"))

    # Load existing progress
    existing = []
    if OUT_FILE.exists():
        existing = json.load(open(OUT_FILE))
    done = len(existing)

    print(f"=== MATH Verified Perturbation ({len(items)} items, {done} done) ===", flush=True)

    for i in range(done, len(items)):
        item = items[i]
        # Generate perturbation with haiku (cheap)
        pert = generate_verified_perturbation(item['question'], item['answer'], item.get('topic', ''))

        if pert is None:
            existing.append({
                'orig_question': item['question'],
                'orig_answer': item['answer'],
                'topic': item.get('topic', ''),
                'pert_question': None,
                'pert_answer': None,
                'verified': False,
                'verify_answer': None,
            })
        else:
            # Verify with sonnet (independent solver)
            verified, verify_ans = verify_perturbation(pert['question'], pert['final_answer'])

            existing.append({
                'orig_question': item['question'],
                'orig_answer': item['answer'],
                'topic': item.get('topic', ''),
                'pert_question': pert['question'],
                'pert_answer': pert.get('solution', ''),
                'pert_final': str(pert['final_answer']),
                'verified': verified,
                'verify_answer': verify_ans,
            })

        if (i + 1) % 10 == 0:
            ok = sum(1 for r in existing if r.get('pert_question') is not None)
            verified_count = sum(1 for r in existing if r.get('verified', False))
            print(f"  {i+1}/{len(items)}: {ok} generated, {verified_count} verified", flush=True)
            # Save periodically
            with open(OUT_FILE, 'w') as f:
                json.dump(existing, f, indent=2)
            time.sleep(0.3)

    # Final save
    with open(OUT_FILE, 'w') as f:
        json.dump(existing, f, indent=2)

    ok = sum(1 for r in existing if r.get('pert_question') is not None)
    verified_count = sum(1 for r in existing if r.get('verified', False))
    print(f"\nDone: {ok}/{len(existing)} generated, {verified_count} verified ({verified_count/max(1,ok)*100:.0f}% verification rate)", flush=True)


if __name__ == "__main__":
    main()
