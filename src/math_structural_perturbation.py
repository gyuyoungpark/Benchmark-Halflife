"""
MATH structural perturbation: rewrite the problem to test the SAME mathematical
concept with a DIFFERENT problem structure. Not just number substitution —
completely different problem setup, different context, different numbers,
but requiring the exact same mathematical skill at the same difficulty.

Goal: fidelity r > 0.8 (if a model truly understands the math, it should
score similarly on both versions; if it memorized the surface form, it won't).
"""
import json, os, sys, re, time
from pathlib import Path
from anthropic import Anthropic

sys.stdout.reconfigure(line_buffering=True)
DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def structural_rewrite(question, solution, topic):
    prompt = f"""You are a math teacher creating an EQUIVALENT problem. Given an original math problem, create a NEW problem that:

1. Tests the EXACT SAME mathematical concept/skill
2. Has COMPLETELY DIFFERENT surface form (different context, scenario, variable names)
3. Uses DIFFERENT numbers (not just substitution — different values entirely)
4. Has the SAME difficulty level
5. Has a DEFINITE numerical or algebraic answer

The new problem should be IMPOSSIBLE to solve by memorizing the original — it must require genuine understanding of the underlying math.

Topic: {topic}

ORIGINAL PROBLEM:
{question}

ORIGINAL SOLUTION:
{solution}

Output ONLY a JSON object:
{{"question": "the new problem text", "solution": "step-by-step solution", "final_answer": "just the answer"}}"""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=1500,
            messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text.strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m: return None
        parsed = json.loads(m.group())
        if "question" in parsed and "final_answer" in parsed:
            return parsed
    except Exception as e:
        print(f"    ERR: {str(e)[:80]}", flush=True)
    return None


def main():
    items = json.load(open(DATA_DIR / "items" / "math_structural_items.json"))
    results = []

    print(f"=== MATH Structural Perturbation ({len(items)} items) ===", flush=True)
    for i, item in enumerate(items):
        pert = structural_rewrite(item['question'], item['answer'], item['topic'])
        results.append({
            'orig_question': item['question'],
            'orig_answer': item['answer'],
            'topic': item['topic'],
            'pert_question': pert['question'] if pert else None,
            'pert_solution': pert.get('solution', '') if pert else None,
            'pert_final': str(pert.get('final_answer', '')) if pert else None,
        })
        if (i + 1) % 20 == 0:
            ok = sum(1 for r in results if r['pert_question'] is not None)
            print(f"  {i+1}/{len(items)} ({ok} ok)", flush=True)
        time.sleep(0.3)

    with open(PERT_DIR / "math_structural_perturbed.json", 'w') as f:
        json.dump(results, f, indent=2)
    ok = sum(1 for r in results if r['pert_question'] is not None)
    print(f"Done: {ok}/{len(results)} successful", flush=True)


if __name__ == "__main__":
    main()
