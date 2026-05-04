"""
#8: Benchmark Half-Life Dashboard — HuggingFace Spaces (Gradio)
Allows users to upload benchmark leaderboard data and compute half-life.
"""
import gradio as gr
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os

def top_k_variance(scores, frac=0.20):
    n = max(1, int(np.ceil(len(scores) * frac)))
    top = np.sort(scores)[-n:]
    return float(top.var())

def fit_decay(t, v):
    if len(t) < 3 or np.all(v == v[0]) or v[0] <= 0:
        return None, None
    try:
        def model(t, a, lam, c):
            return a * np.exp(-lam * t) + c
        p0 = [max(v[0]-v[-1], v[0]*0.1), 0.05, min(v)]
        bounds = ([0, 1e-6, 0], [v.max()*10+1e-6, 2.0, v.max()+1e-6])
        popt, _ = curve_fit(model, t, v, p0=p0, bounds=bounds, maxfev=20000)
        if popt[1] < 1e-5:
            return None, None
        tau = float(np.log(2) / popt[1])
        return tau, popt
    except:
        return None, None

def bootstrap_ci(df, top_k=0.20, n_boot=500):
    quarters = sorted(df['quarter'].unique())
    rng = np.random.RandomState(42)
    taus = []
    for _ in range(n_boot):
        t_mo, vars_ = [], []
        for q in quarters:
            g = df[df['quarter'] == q]
            if len(g) < 10: continue
            idx = rng.randint(0, len(g), size=len(g))
            boot_scores = g.iloc[idx]['score'].values
            v = top_k_variance(boot_scores, top_k)
            t_mo.append((q - quarters[0]).n * 3.0)
            vars_.append(v)
        tau, _ = fit_decay(np.array(t_mo), np.array(vars_))
        if tau and 0.1 < tau < 500:
            taus.append(tau)
    if not taus:
        return None, None, None
    return float(np.median(taus)), float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))

def analyze(file, top_k):
    if file is None:
        return "Please upload a CSV file.", None

    df = pd.read_csv(file.name)

    # Auto-detect columns
    date_col = None
    for c in ['eval_date', 'date', 'release_date', 'submission_date']:
        if c in df.columns:
            date_col = c; break
    score_col = None
    for c in ['score', 'accuracy', 'acc', 'value']:
        if c in df.columns:
            score_col = c; break

    if not date_col or not score_col:
        return f"Cannot find date/score columns. Found: {list(df.columns)}", None

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[score_col])
    df['quarter'] = df[date_col].dt.to_period('Q')

    quarters = sorted(df['quarter'].unique())
    t_mo, vars_ = [], []
    for q in quarters:
        g = df[df['quarter'] == q]
        if len(g) < 10: continue
        v = top_k_variance(g[score_col].values, top_k)
        t_mo.append((q - quarters[0]).n * 3.0)
        vars_.append(v)

    t_mo, vars_ = np.array(t_mo), np.array(vars_)
    tau, popt = fit_decay(t_mo, vars_)
    tau_med, ci_lo, ci_hi = bootstrap_ci(df, top_k)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_mo, vars_, 'o-', color='#1976D2', linewidth=2, markersize=8, zorder=3)
    if tau and popt is not None:
        t_smooth = np.linspace(0, max(t_mo)*1.2, 50)
        ax.plot(t_smooth, popt[0]*np.exp(-popt[1]*t_smooth)+popt[2], '--', color='red', alpha=0.7,
                label=f'τ½ = {tau:.1f} mo')
    ax.set_xlabel('Months from first quarter')
    ax.set_ylabel(f'Top-{int(top_k*100)}% variance')
    ax.set_title('Discriminative Half-Life Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name, dpi=150)
    plt.close()

    # Result text
    if tau:
        result = f"""## Discriminative Half-Life

**τ½ = {tau:.1f} months**

Bootstrap 95% CI: [{ci_lo:.1f}, {ci_hi:.1f}] months (500 resamples)

### Details
- Quarters analyzed: {len(t_mo)}
- Unique models: {df['model'].nunique() if 'model' in df.columns else len(df)}
- Top-{int(top_k*100)}% stratum
- Initial variance: {vars_[0]:.5f}
- Final variance: {vars_[-1]:.5f}
- Decay: {(1 - vars_[-1]/vars_[0])*100:.0f}%
"""
    else:
        result = """## No Decay Detected

The benchmark may still be in its honeymoon phase, or the data
does not show a consistent decay pattern.
"""

    return result, tmp.name

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="Upload benchmark CSV (columns: model, score, eval_date)"),
        gr.Slider(0.05, 0.50, value=0.20, step=0.05, label="Top-k fraction"),
    ],
    outputs=[
        gr.Markdown(label="Result"),
        gr.Image(label="Decay Plot"),
    ],
    title="🔬 Benchmark Half-Life Calculator",
    description="Upload a CSV with columns `model`, `score`, `eval_date` to compute the discriminative half-life of your LLM benchmark.",
    examples=[],
)

if __name__ == "__main__":
    demo.launch()
