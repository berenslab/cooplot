from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# Optional circos
try:
    from mne_connectivity.viz import plot_connectivity_circle as _circle

    HAVE_CIRCLE = True
except Exception:
    HAVE_CIRCLE = False


def _casefold(s: str) -> str:
    # robust, locale-agnostic lowercase (handles ß, accents, etc.)
    return (s or "").casefold()


def _split_name(name: str) -> Tuple[str, str]:
    # returns (last, first) in a simple way; good enough for sorting
    parts = (name or "").strip().split()
    if not parts:
        return "", ""
    last = parts[-1]
    first = " ".join(parts[:-1])
    return _casefold(last), _casefold(first)


def _group_value_for(name: str, people: List[dict], group_col: Optional[str]) -> str:
    if not group_col:
        return ""
    for p in people:
        if p.get("name") == name:
            return str(p.get(group_col, "") or "")
    return ""


def _order_indices(
    labels: List[str], people: List[dict], group_col: Optional[str]
) -> Tuple[List[int], List[str]]:
    """
    Compute a permutation that orders labels by:
      - group value (casefolded) if group_col is given,
      - then last name, then first name (both casefolded).
    Returns (perm_indices, ordered_labels).
    """
    # Prepare sortable keys
    keys = []
    for i, name in enumerate(labels):
        g = _group_value_for(name, people, group_col)
        last, first = _split_name(name)
        keys.append((i, _casefold(g), last, first))

    # Sort by (group, last, first); if group_col is None, group key is ""
    keys.sort(key=lambda t: (t[1], t[2], t[3]))

    perm = [t[0] for t in keys]
    ordered = [labels[i] for i in perm]
    return perm, ordered


# --- existing palette / color utilities remain unchanged ---------------------


def _to_hex(rgb_or_hex) -> str:
    if isinstance(rgb_or_hex, str) and rgb_or_hex.startswith("#"):
        return rgb_or_hex
    r, g, b = rgb_or_hex
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def _auto_palette(keys: List[str]) -> Dict[str, str]:
    cmap = plt.cm.get_cmap("tab20")
    uniq = sorted(set(keys))
    return {k: mcolors.to_hex(cmap(i % cmap.N)) for i, k in enumerate(uniq)}


def _reds_shaded():
    original = plt.cm.Reds
    start, stop, n = 0.20, 0.80, 256
    reduced = original(np.linspace(start, stop, n))
    reduced[0] = np.array([1, 1, 1, 1])
    return mcolors.LinearSegmentedColormap.from_list("white_to_darker_Reds", reduced)


def _node_colors(
    labels: List[str],
    people: List[dict],
    group_col: Optional[str],
    palette: Optional[Dict[str, str]],
):
    # build group list in label order
    name_to_group = {}
    for p in people:
        name_to_group[p.get("name")] = p.get(group_col, "") if group_col else ""
    keys = [name_to_group.get(n, "") for n in labels]
    pal = palette or (_auto_palette(keys) if group_col else {"": "#808080"})
    return [pal.get(k, "#808080") for k in keys], pal


def plot_panels(
    mats: Dict[str, dict],
    *,
    out_path: str | Path | None,
    people: List[dict],
    group_col: Optional[str] = None,
    palette: Optional[Dict[str, str]] = None,
    vmax: Optional[int] = None,
    style: str = "circle",
    save_pdf: bool = False,
    show: bool = False,
    return_fig: bool = False,
    cap_weights: Optional[int] = None,
    counts_label: str = "Shared coauthorships",
    legend_counts: bool = True,
    legend_groups: bool = True,
) -> Optional[plt.Figure]:
    wins = list(mats.keys())
    if not wins:
        raise ValueError("No matrices to plot.")

    base_labels = mats[wins[0]]["labels"]
    # Compute one consistent permutation for ALL windows
    perm, ordered_labels = _order_indices(base_labels, people, group_col)

    # Determine global vmax and apply cap if requested
    vmax_base = vmax or max(int(np.max(np.array(mats[w]["matrix"]))) for w in wins)
    vmax_used = min(vmax_base, cap_weights) if cap_weights is not None else vmax_base

    cmap = _reds_shaded()
    colors, used_palette = _node_colors(ordered_labels, people, group_col, palette)

    # Build group keys in label order for a legend
    name_to_group = {
        p.get("name"): (p.get(group_col, "") if group_col else "") for p in people
    }
    group_keys_in_order = [name_to_group.get(n, "") for n in ordered_labels]
    # stable unique
    unique_groups = []
    for k in group_keys_in_order:
        if k not in unique_groups:
            unique_groups.append(k)

    vmax_all = vmax or max(int(np.max(np.array(mats[w]["matrix"]))) for w in wins)
    cmap = _reds_shaded()
    colors, used_palette = _node_colors(ordered_labels, people, group_col, palette)

    # Heatmap path
    if style == "heatmap" or not HAVE_CIRCLE:
        fig, axes = plt.subplots(
            1, len(wins), figsize=(6 * len(wins), 6), squeeze=False
        )
        axs = axes.ravel().tolist()
        last_im = None

        for i, w in enumerate(wins):
            M = np.array(mats[w]["matrix"], dtype=int)
            # apply permutation
            M = M[np.ix_(perm, perm)]
            # cap for plotting if requested
            if cap_weights is not None:
                M = np.minimum(M, cap_weights)

            ax = axs[i]
            last_im = ax.imshow(
                M, vmin=0, vmax=vmax_used, cmap=cmap, interpolation="nearest"
            )
            ax.set_title(w, loc="left")
            ax.set_xticks(range(len(ordered_labels)))
            ax.set_xticklabels(ordered_labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(ordered_labels)))
            ax.set_yticklabels(ordered_labels, fontsize=7)

        # ---- counts legend (colorbar)
        if legend_counts:
            # Use a ScalarMappable so we control the label
            norm = mcolors.Normalize(vmin=0, vmax=vmax_used)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02)
            label = f" (capped at {cap_weights}" if cap_weights is not None else ""
            cb.set_label(label)
            cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # ---- groups legend (node color legend)
        if legend_groups and group_col:
            handles = [
                Patch(
                    facecolor=used_palette.get(g, "#808080"),
                    edgecolor="none",
                    label=g or "Unlabeled",
                )
                for g in unique_groups
            ]
            # leave some bottom space for legend
            fig.subplots_adjust(bottom=0.18)
            fig.legend(
                handles=handles,
                title=group_col,
                loc="lower center",
                ncol=min(len(handles), 6),
                frameon=False,
            )

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            if save_pdf and str(out_path).lower().endswith(".png"):
                fig.savefig(
                    Path(out_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight"
                )
        if show and not out_path:
            plt.show()
        if return_fig:
            return fig
        if out_path and not return_fig:
            plt.close(fig)
        return None

    # Circos path
    fig, axes = plt.subplots(
        1,
        len(wins),
        subplot_kw={"projection": "polar"},
        figsize=(9 * len(wins), 9),
        facecolor="white",
    )
    if len(wins) == 1:
        axes = [axes]
    axs = axes
    # Leave margin for the colorbar and optional group legend
    right_margin = 0.88
    bottom_margin = 0.18 if legend_groups and group_col else 0.08
    fig.subplots_adjust(right=right_margin, bottom=bottom_margin)

    for i, w in enumerate(wins):
        M = np.array(mats[w]["matrix"], dtype=int)
        M = M[np.ix_(perm, perm)]
        if cap_weights is not None:
            M = np.minimum(M, cap_weights)

        _circle(
            M,
            node_names=ordered_labels,
            node_colors=colors,
            vmin=0,
            vmax=vmax_used,
            colorbar=False,  # we'll add our own CB
            facecolor="white",
            textcolor="black",
            colormap=cmap,
            node_edgecolor="white",
            fig=fig,
            ax=axs[i],
            show=False,
            fontsize_title=18,
            fontsize_names=10,
        )
        axs[i].set_title(w, fontsize=18, loc="left", pad=20, color="black")
        # label backgrounds
        for j, label in enumerate(axs[i].texts):
            label.set_color("white")
            label.set_bbox(dict(facecolor=colors[j], edgecolor=colors[j], pad=0.4))
            rot = label.get_rotation()
            if 90 <= rot < 270:
                label.set_rotation(rot - 180)
                label.set_va("center")
                label.set_ha("left")

    # ---- counts legend (colorbar)
    if legend_counts:
        norm = mcolors.Normalize(vmin=0, vmax=vmax_used)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.canvas.draw()
        ax_positions = [ax.get_position() for ax in axs]
        right_edge = max(pos.x1 for pos in ax_positions)
        bottom_edge = min(pos.y0 for pos in ax_positions)
        top_edge = max(pos.y1 for pos in ax_positions)
        available_width = max(1.0 - right_edge, 1e-3)
        cbar_width = min(0.02, available_width * 0.9)
        cbar_height = max((top_edge - bottom_edge) * 0.6, 0.05)
        y0 = bottom_edge + (top_edge - bottom_edge - cbar_height) / 2
        x0 = right_edge + (available_width - cbar_width) / 2
        cax = fig.add_axes([x0, y0, cbar_width, cbar_height])
        cb = fig.colorbar(sm, cax=cax)
        label = counts_label + (
            f" (capped at {cap_weights})" if cap_weights is not None else ""
        )
        cb.set_label(label)
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cb.ax.tick_params(labelsize=8)

    # ---- groups legend
    if legend_groups and group_col:
        handles = [
            Patch(
                facecolor=used_palette.get(g, "#808080"),
                edgecolor="none",
                label=g or "Unlabeled",
            )
            for g in unique_groups
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            frameon=False,
        )

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if save_pdf and str(out_path).lower().endswith(".png"):
            fig.savefig(
                Path(out_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight"
            )
    if show and not out_path:
        plt.show()
    if return_fig:
        return fig
    if out_path and not return_fig:
        plt.close(fig)
    return None
