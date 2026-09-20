"""Render the hand-edited E3 adjacency (models/graph_edges_manual.txt)
over each class-mean shape, for manual review. Edge parsing comes from
core.graph (per AGENTS.md, don't reimplement core logic in scripts); only
the legend-label parsing (review-only, doesn't belong in core.graph) stays
local to this script.

Usage: uv run python scripts/graph_review_manual.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from catface.core.graph import DEFAULT_EDGES_PATH, build_cat_edges

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "e3"

LEGEND_RE = re.compile(r"^#\s+(\d+):\s*(.+)$")


def parse_legend(path: Path) -> dict[int, str]:
    groups: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if m := LEGEND_RE.match(line):
            idx, label = m.groups()
            groups[int(idx)] = label.split(",")[0].strip()
    return groups


def write_png(shape: np.ndarray, edges: list[tuple[int, int]], groups: dict[int, str], out_path) -> None:
    distinct_groups = sorted(set(groups.values()))
    cmap = plt.get_cmap("tab10", len(distinct_groups))
    color_by_group = {g: cmap(i) for i, g in enumerate(distinct_groups)}

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, j in edges:
        ax.plot([shape[i, 0], shape[j, 0]], [shape[i, 1], shape[j, 1]], color="0.6", linewidth=1, zorder=1)

    for i, (x, y) in enumerate(shape):
        ax.scatter(x, y, s=250, color=color_by_group.get(groups.get(i), "gray"), zorder=2, edgecolors="black")
        ax.annotate(str(i), (x, y), fontsize=11, fontweight="bold", ha="center", va="center", zorder=3)

    ax.invert_yaxis()  # landmarks are in image (y-down) coordinates
    ax.set_aspect("equal")
    ax.set_title('E3 manual adjacency review over "relaxed" class mean shape')
    ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=12, label=g)
               for g, c in color_by_group.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    groups = parse_legend(DEFAULT_EDGES_PATH)
    edges = build_cat_edges(DEFAULT_EDGES_PATH)
    class_means = json.loads((ROOT / "models" / "class_means.json").read_text())

    for name in ["attentive", "relaxed", "uncomfortable"]:
        shape = np.array(class_means[name])
        savepath = OUT_DIR / f'graph_{name}_manual_review.png'
        write_png(shape, edges, groups, savepath)
        print(f"wrote {savepath}")


if __name__ == "__main__":
    main()
