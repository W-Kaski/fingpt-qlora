"""Render side-by-side qualitative comparisons."""

from IPython.display import display, HTML


def render_comparison_table(results: list[dict], n_examples: int = 10) -> str:
    """Render an HTML table comparing base vs fine-tuned outputs."""
    rows = []
    for i, r in enumerate(results[:n_examples]):
        task = r.get("task_type", "unknown")
        ref = r["reference"][:300].replace("\n", "<br>")
        pred_base = r.get("base_prediction", "")[:300].replace("\n", "<br>")
        pred_ft = r.get("ft_prediction", "")[:300].replace("\n", "<br>")

        rows.append(f"""
        <tr>
            <td>{i+1}</td>
            <td>{task}</td>
            <td>{ref}</td>
            <td style="background:#fff3f3">{pred_base}</td>
            <td style="background:#f3fff3">{pred_ft}</td>
        </tr>
        """)

    table = f"""
    <table border="1" style="border-collapse:collapse; width:100%; font-size:12px">
    <tr>
        <th>#</th><th>Task</th><th>Reference</th>
        <th style="background:#ffe0e0">Base Model</th>
        <th style="background:#e0ffe0">Fine-tuned</th>
    </tr>
    {''.join(rows)}
    </table>
    """
    return table


def show_comparison(results: list[dict], n_examples: int = 10):
    """Display comparison table in Jupyter notebook."""
    html = render_comparison_table(results, n_examples)
    display(HTML(html))


def select_notable_examples(
    results: list[dict],
    n_improved: int = 5,
    n_regressed: int = 2,
) -> dict:
    """Select examples showing most improvement and regression."""
    scored = []
    for r in results:
        base = r.get("base_prediction", "")
        ft = r.get("ft_prediction", "")
        ref = r.get("reference", "")

        base_words = set(base.lower().split())
        ft_words = set(ft.lower().split())
        ref_words = set(ref.lower().split())
        base_score = len(ref_words & base_words)
        ft_score = len(ref_words & ft_words)
        improvement = ft_score - base_score
        scored.append({**r, "improvement": improvement})

    scored.sort(key=lambda x: x["improvement"], reverse=True)

    return {
        "most_improved": scored[:n_improved],
        "most_regressed": scored[-n_regressed:],
    }
