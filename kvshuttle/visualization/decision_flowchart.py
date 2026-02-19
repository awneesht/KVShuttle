"""Generate the KVShuttle decision flowchart for practitioners."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, wrap=True,
            bbox=dict(facecolor="none", edgecolor="none"))


def _draw_diamond(ax, x, y, w, h, text, color="#FFF3CD", fontsize=8.5):
    """Draw a diamond (decision node) with centered text."""
    diamond = plt.Polygon(
        [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold")


def _arrow(ax, x1, y1, x2, y2, label="", label_pos="mid"):
    """Draw an arrow with an optional label positioned to avoid overlap."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5),
    )
    if label:
        if label_pos == "start":
            lx, ly = x1 + (x2 - x1) * 0.15, y1 + (y2 - y1) * 0.15
        else:
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
        # For horizontal arrows, place label above; for vertical, place right
        if abs(y2 - y1) < 0.1:  # horizontal
            ax.text(lx, ly + 0.15, label, fontsize=8, fontweight="bold",
                    color="#444444", ha="center", va="bottom")
        else:  # vertical
            ax.text(lx + 0.15, ly, label, fontsize=8, fontweight="bold",
                    color="#444444", ha="left", va="center")


def generate_decision_flowchart(output_path: str | Path | None = None) -> None:
    """Generate and save the KVShuttle decision flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 13))
    ax.set_xlim(-5.5, 6)
    ax.set_ylim(-1.5, 11.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors
    START = "#E3F2FD"
    DECISION = "#FFF3CD"
    RECOMMEND = "#D4EDDA"
    WARN = "#F8D7DA"

    # Layout constants
    DX = 0  # diamond center x
    RX = 4.5  # recommendation box center x
    BW = 2.4  # box width
    BH = 0.85  # box height

    # ── Title ──
    ax.text(0, 11.0, "KVShuttle Decision Flowchart",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, 10.6,
            "Which KV cache compressor should I use for disaggregated serving?",
            ha="center", va="center", fontsize=10, color="#666666", style="italic")

    # ── Start box ──
    _draw_box(ax, DX, 9.8, 3.6, 0.6, "KV cache transfer\nbetween prefill & decode GPUs",
              START, fontsize=9, bold=True)

    # ── Decision 1: Bandwidth ──
    _arrow(ax, DX, 9.5, DX, 8.85)
    _draw_diamond(ax, DX, 8.4, 3.6, 0.9, "Network bandwidth\n> 50 Gbps?", DECISION)

    _arrow(ax, 1.8, 8.4, RX - BW / 2, 8.4, "Yes", "start")
    _draw_box(ax, RX, 8.4, BW, BH,
              "Don't compress\n(identity)\n1.0x | TA=1.00",
              RECOMMEND, fontsize=8.5, bold=True)

    _arrow(ax, DX, 7.95, DX, 7.35, "No")

    # ── Decision 2: High quality ──
    _draw_diamond(ax, DX, 6.9, 3.6, 0.9, "Need high quality?\n(token agree > 0.95)", DECISION)

    _arrow(ax, 1.8, 6.9, RX - BW / 2, 6.9, "Yes", "start")
    _draw_box(ax, RX, 6.9, BW, BH,
              "uniform_int8\n2.0x | TA=0.97\nppl\u0394=0.03",
              RECOMMEND, fontsize=8.5, bold=True)

    _arrow(ax, DX, 6.45, DX, 5.85, "No")

    # ── Decision 3: Moderate quality ──
    _draw_diamond(ax, DX, 5.4, 3.6, 0.9, "Moderate quality OK?\n(token agree > 0.90)", DECISION)

    _arrow(ax, 1.8, 5.4, RX - BW / 2, 5.4, "Yes", "start")
    _draw_box(ax, RX, 5.4, BW, BH,
              "cachegen\n3.5x | TA=0.93\nppl\u0394=0.81",
              RECOMMEND, fontsize=8.5, bold=True)

    _arrow(ax, DX, 4.95, DX, 4.35, "No")

    # ── Decision 4: Max compression ──
    _draw_diamond(ax, DX, 3.9, 3.6, 0.9,
                  "Bandwidth < 10 Gbps?\n(need max compression)", DECISION)

    _arrow(ax, 1.8, 3.9, RX - BW / 2, 3.9, "Yes", "start")
    _draw_box(ax, RX, 3.9, BW, BH,
              "kivi_2bit\n6.5x | TA=0.74\nppl\u0394=7.5",
              "#FFF3CD", fontsize=8.5, bold=True)

    _arrow(ax, DX, 3.45, DX, 2.75, "No")
    _draw_box(ax, DX, 2.3, 3.0, 0.85,
              "fp8_e4m3\n2.0x | HW-accelerated\nBest if GPU supports FP8",
              RECOMMEND, fontsize=8.5, bold=True)

    # ── Avoid box (left side) ──
    _draw_box(ax, -3.8, 5.4, 2.4, 1.8,
              "Avoid for\ngeneration tasks:\n\nuniform_int4\nTA=0.59 | ppl\u0394=13k\n\ncascade\nTA=0.33 | ppl\u0394=18k",
              WARN, fontsize=7.5, bold=False)

    # ── Legend ──
    y_leg = 0.8
    ax.text(-4.5, y_leg + 0.3, "Legend", fontsize=9, fontweight="bold")
    ax.text(-4.5, y_leg, "TA = Token Agreement (greedy decode match rate)", fontsize=7.5, color="#555")
    ax.text(-4.5, y_leg - 0.3, "ppl\u0394 = Perplexity delta vs uncompressed", fontsize=7.5, color="#555")
    ax.text(-4.5, y_leg - 0.6, "N.Nx = Compression ratio", fontsize=7.5, color="#555")
    ax.text(-4.5, y_leg - 0.9, "Break-even: GPU pipelined, Tesla T4", fontsize=7.5, color="#555")
    ax.text(-4.5, y_leg - 1.2, "Quality: FP16, 5 models, 50 WikiText prompts", fontsize=7.5, color="#555")

    plt.tight_layout()

    if output_path is None:
        output_path = Path("paper/figures/decision_flowchart.pdf")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved decision flowchart to {output_path}")


if __name__ == "__main__":
    generate_decision_flowchart()
