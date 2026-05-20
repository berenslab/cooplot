from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    MaxNLocator,
)


def _circle_layout(
    n: int,
    weights: Optional[np.ndarray],
    *,
    gap_frac: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centers, widths) in radians for ``n`` nodes around the circle.

    ``weights`` controls relative arc length per node. ``None`` (or any
    non-positive/invalid total) yields a uniform layout. ``gap_frac`` is the
    fraction of the full circle reserved for gaps between bars.
    """
    if n <= 0:
        return np.empty(0), np.empty(0)
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,) or not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(n, dtype=float)
    w = w / w.sum()
    total_gap = min(max(gap_frac, 0.0), 0.5) * 2 * np.pi
    arc = max(0.0, 2 * np.pi - total_gap)
    widths = w * arc
    gap = (2 * np.pi - widths.sum()) / n
    centers = np.cumsum(widths) - widths / 2 + np.arange(n) * gap
    return centers, widths


def _draw_circle(
    ax,
    M: np.ndarray,
    labels: List[str],
    node_colors: List[str],
    *,
    cmap,
    vmin: float,
    vmax: float,
    weights: Optional[np.ndarray] = None,
    rotate: float = 0.0,
    fontsize_names: float = 8.0,
    label_box_pad: float = 0.4,
    connection_linewidth: float = 1.5,
    node_height: float = 1.0,
    node_linewidth: float = 2.0,
    node_edgecolor: str = "white",
    padding: float = 6.0,
) -> None:
    """Draw a co-occurrence circle plot onto a polar axes.

    Replaces the previous ``mne_connectivity.plot_connectivity_circle`` path.
    Bars sit at radius 9..10; labels sit just outside; connection beziers curve
    through r=5 (matches the prior geometry).
    """
    n = len(labels)
    if n == 0:
        return

    centers, widths = _circle_layout(n, weights)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 10 + padding)
    ax.spines["polar"].set_visible(False)
    if rotate:
        ax.set_theta_offset(np.deg2rad(rotate))

    vrange = (vmax - vmin) if vmax > vmin else 1.0

    tril_i, tril_j = np.tril_indices(n, -1)
    vals = M[tril_i, tril_j]
    keep = vals > 0
    tril_i, tril_j, vals = tril_i[keep], tril_j[keep], vals[keep]
    # Draw weakest first so strongest land on top
    order = np.argsort(np.abs(vals))
    tril_i, tril_j, vals = tril_i[order], tril_j[order], vals[order]

    for ii, jj, v in zip(tril_i, tril_j, vals):
        t0, t1 = centers[ii], centers[jj]
        path = mpath.Path(
            [(t0, 10), (t0, 5), (t1, 5), (t1, 10)],
            [
                mpath.Path.MOVETO,
                mpath.Path.CURVE4,
                mpath.Path.CURVE4,
                mpath.Path.LINETO,
            ],
        )
        ax.add_patch(
            mpatches.PathPatch(
                path,
                fill=False,
                edgecolor=cmap((float(v) - vmin) / vrange),
                linewidth=connection_linewidth,
                alpha=1.0,
            )
        )

    bars = ax.bar(
        centers,
        np.full(n, node_height),
        width=widths,
        bottom=9,
        edgecolor=node_edgecolor,
        lw=node_linewidth,
        align="center",
    )
    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    label_r = 9.4 + node_height
    angles_deg = np.rad2deg(centers)
    for name, theta, deg, color in zip(labels, centers, angles_deg, node_colors):
        screen_angle = (deg + rotate) % 360
        if 90 <= screen_angle < 270:
            text_rot = screen_angle - 180
            ha = "right"
        else:
            text_rot = screen_angle
            ha = "left"
        ax.text(
            theta,
            label_r,
            name,
            rotation=text_rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            size=fontsize_names,
            color="white",
            bbox=dict(facecolor=color, edgecolor=color, pad=label_box_pad),
        )


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


def _group_value_for(name: str, label_to_group: Optional[Dict[str, str]]) -> str:
    if not label_to_group:
        return ""
    return str(label_to_group.get(name, "") or "")


def _order_indices(
    labels: List[str], label_to_group: Optional[Dict[str, str]]
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
        g = _group_value_for(name, label_to_group)
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
    label_to_group: Optional[Dict[str, str]],
    palette: Optional[Dict[str, str]],
):
    label_to_group = label_to_group or {}
    keys = [label_to_group.get(n, "") for n in labels]
    has_groups = any(k for k in keys)
    if palette:
        pal = palette
    elif has_groups:
        pal = _auto_palette(keys)
    else:
        pal = {"": "#808080"}
    default_color = pal.get("", "#808080")
    colors = [pal.get(k, default_color) for k in keys]
    return colors, pal


def plot_panels(
    mats: Dict[str, dict],
    *,
    out_path: str | Path | None,
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
    heatmap_counts: bool = False,
    figsize: Optional[tuple] = None,
    rotate: float = 0.0,
    bin_width: str = "uniform",
) -> Optional[plt.Figure]:
    wins = list(mats.keys())
    if not wins:
        raise ValueError("No matrices to plot.")

    style_requested = (style or "circle").lower()
    allowed_styles = {"circle", "heatmap", "both"}
    if style_requested not in allowed_styles:
        raise ValueError(
            f"Unsupported style '{style}'. Expected one of {sorted(allowed_styles)}."
        )
    style_mode = style_requested
    if style_mode == "both" and len(wins) != 1:
        raise ValueError("Style 'both' is only supported when a single window is provided.")

    bin_width_mode = (bin_width or "uniform").lower()
    allowed_bin_widths = {"uniform", "total"}
    if bin_width_mode not in allowed_bin_widths:
        raise ValueError(
            f"Unsupported bin_width '{bin_width}'. Expected one of {sorted(allowed_bin_widths)}."
        )

    base_labels = mats[wins[0]]["labels"]
    real_label_to_group = mats[wins[0]].get("label_to_group", {}) or {}
    has_real_groups = bool(real_label_to_group)

    # Order using only real group info so the per-author fallback below doesn't
    # promote first-name into the primary sort key.
    perm, ordered_labels = _order_indices(base_labels, real_label_to_group)

    if has_real_groups:
        label_to_group_map = real_label_to_group
        group_col_for_plot = group_col or "Group"
    else:
        # No group info baked into mats: treat each author as their own group
        # so colors are distinct per author. Only render a legend when the user
        # explicitly opts in (e.g. group_col="name"), since names already label
        # the axes.
        label_to_group_map = {label: label for label in base_labels}
        group_col_for_plot = group_col

    # Determine global vmax and apply cap if requested
    vmax_base = vmax or max(int(np.max(np.array(mats[w]["matrix"]))) for w in wins)
    vmax_used = min(vmax_base, cap_weights) if cap_weights is not None else vmax_base

    cmap = _reds_shaded()
    node_colors, used_palette = _node_colors(
        ordered_labels,
        label_to_group_map,
        palette,
    )

    # Build group keys in label order for a legend
    group_keys_in_order = [label_to_group_map.get(n, "") for n in ordered_labels]
    # stable unique
    unique_groups = []
    for k in group_keys_in_order:
        if k not in unique_groups:
            unique_groups.append(k)

    if style_mode == "both":
        default_figsize = (15, 9)
    elif style_mode == "heatmap":
        default_figsize = (6 * len(wins), 6)
    else:
        default_figsize = (9 * len(wins), 9)
    figsize_used = default_figsize if figsize is None else figsize
    scale_x = figsize_used[0] / default_figsize[0] if default_figsize[0] else 1.0
    scale_y = figsize_used[1] / default_figsize[1] if default_figsize[1] else 1.0
    scale = min(scale_x, scale_y)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    def _scaled(value: float, floor: float) -> float:
        return float(max(floor, value * scale))

    tick_font = _scaled(7, 4)
    cb_tick_font = _scaled(8, 4)
    cb_label_font = _scaled(10, 6)
    legend_font = _scaled(9, 6)

    legend_title_font = _scaled(10, 7)
    heatmap_title_font = _scaled(12, 9)
    tick_pad = max(1.0, 3.0 * scale)
    tick_box_pad = max(0.2, 0.2 * scale)
    count_font = _scaled(6, 4)
    stroke_width = max(0.6, 0.8 * scale)
    circle_title_font = _scaled(18, 10)
    circle_name_font = _scaled(10, 6)
    label_box_pad = max(0.3, 0.4 * scale)

    # Heatmap path
    if style_mode == "heatmap":
        fig, axes = plt.subplots(
            1,
            len(wins),
            figsize=figsize_used,
            squeeze=False,
        )
        axs = axes.ravel().tolist()
        last_im = None
        right_margin = 0.88 if legend_counts else 0.96
        bottom_margin = 0.25 if legend_groups and group_col_for_plot else 0.15

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
            ax.set_title(w, loc="left", fontsize=heatmap_title_font)
            tick_positions = np.arange(len(ordered_labels))
            x_locator = FixedLocator(tick_positions)
            y_locator = FixedLocator(tick_positions)
            ax.xaxis.set_major_locator(x_locator)
            ax.yaxis.set_major_locator(y_locator)
            ax.xaxis.set_major_formatter(FixedFormatter(ordered_labels))
            ax.yaxis.set_major_formatter(FixedFormatter(ordered_labels))
            ax.tick_params(
                axis="x", labelsize=tick_font, labelrotation=90, pad=tick_pad
            )
            ax.tick_params(axis="y", labelsize=tick_font, pad=tick_pad)

            # match tick label styling to group colors, similar to the circle plot
            for tick_label, color in zip(ax.get_xticklabels(), node_colors):
                tick_label.set_color("white")
                tick_label.set_bbox(
                    dict(facecolor=color, edgecolor=color, pad=tick_box_pad)
                )
                tick_label.set_rotation_mode("anchor")
                tick_label.set_ha("right")
                tick_label.set_va("center")

            for tick_label, color in zip(ax.get_yticklabels(), node_colors):
                tick_label.set_color("white")
                tick_label.set_bbox(
                    dict(facecolor=color, edgecolor=color, pad=tick_box_pad)
                )
                tick_label.set_va("center")
                tick_label.set_ha("right")

            if heatmap_counts:
                vmax_for_text = float(vmax_used) if vmax_used else 0.0
                threshold = vmax_for_text * 0.5
                for r in range(M.shape[0]):
                    for c in range(M.shape[1]):
                        val = M[r, c]
                        try:
                            val_float = float(val)
                        except Exception:
                            continue
                        if np.isnan(val_float):
                            continue
                        try:
                            disp = int(round(val_float))
                        except Exception:
                            disp = val_float
                        if disp == 0:
                            continue
                        color_choice = "black"
                        if vmax_for_text > 0 and val_float >= threshold:
                            color_choice = "white"
                        txt = ax.text(
                            c,
                            r,
                            f"{disp}",
                            ha="center",
                            va="center",
                            fontsize=count_font,
                            color=color_choice,
                        )
                        stroke = "black" if color_choice == "white" else "white"
                        txt.set_path_effects(
                            [
                                patheffects.withStroke(
                                    linewidth=stroke_width, foreground=stroke
                                )
                            ]
                        )

        fig.tight_layout()
        fig.subplots_adjust(right=right_margin, bottom=bottom_margin)

        # ---- counts legend (colorbar)
        if legend_counts:
            # Use a ScalarMappable so we control the label
            norm = mcolors.Normalize(vmin=0, vmax=vmax_used)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.04)
            label = counts_label + (
                f" (capped at {cap_weights})" if cap_weights is not None else ""
            )
            cb.set_label(label, fontsize=cb_label_font)
            cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            cb.ax.tick_params(labelsize=cb_tick_font)

        # ---- groups legend (node color legend)
        if legend_groups and group_col_for_plot:
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
                prop={"size": legend_font},
                title_fontsize=legend_title_font,
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

    if style_mode == "both":
        fig = plt.figure(figsize=figsize_used)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
        ax_heat = fig.add_subplot(gs[0, 0])
        ax_circle = fig.add_subplot(gs[0, 1], projection="polar")
        axs = [ax_heat, ax_circle]
        w = wins[0]
        M = np.array(mats[w]["matrix"], dtype=int)
        M = M[np.ix_(perm, perm)]
        if cap_weights is not None:
            M = np.minimum(M, cap_weights)

        ax_heat.imshow(M, vmin=0, vmax=vmax_used, cmap=cmap, interpolation="nearest")
        ax_heat.set_title(w, loc="left", fontsize=heatmap_title_font)
        tick_positions = np.arange(len(ordered_labels))
        x_locator = FixedLocator(tick_positions)
        y_locator = FixedLocator(tick_positions)
        ax_heat.xaxis.set_major_locator(x_locator)
        ax_heat.yaxis.set_major_locator(y_locator)
        ax_heat.xaxis.set_major_formatter(FixedFormatter(ordered_labels))
        ax_heat.yaxis.set_major_formatter(FixedFormatter(ordered_labels))
        ax_heat.tick_params(axis="x", labelsize=tick_font, labelrotation=90, pad=tick_pad)
        ax_heat.tick_params(axis="y", labelsize=tick_font, pad=tick_pad)

        for tick_label, color in zip(ax_heat.get_xticklabels(), node_colors):
            tick_label.set_color("white")
            tick_label.set_bbox(
                dict(facecolor=color, edgecolor=color, pad=tick_box_pad)
            )
            tick_label.set_rotation_mode("anchor")
            tick_label.set_ha("right")
            tick_label.set_va("center")

        for tick_label, color in zip(ax_heat.get_yticklabels(), node_colors):
            tick_label.set_color("white")
            tick_label.set_bbox(
                dict(facecolor=color, edgecolor=color, pad=tick_box_pad)
            )
            tick_label.set_va("center")
            tick_label.set_ha("right")

        if heatmap_counts:
            vmax_for_text = float(vmax_used) if vmax_used else 0.0
            threshold = vmax_for_text * 0.5
            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    val = M[r, c]
                    try:
                        val_float = float(val)
                    except Exception:
                        continue
                    if np.isnan(val_float):
                        continue
                    try:
                        disp = int(round(val_float))
                    except Exception:
                        disp = val_float
                    if disp == 0:
                        continue
                    color_choice = "black"
                    if vmax_for_text > 0 and val_float >= threshold:
                        color_choice = "white"
                    txt = ax_heat.text(
                        c,
                        r,
                        f"{disp}",
                        ha="center",
                        va="center",
                        fontsize=count_font,
                        color=color_choice,
                    )
                    stroke = "black" if color_choice == "white" else "white"
                    txt.set_path_effects(
                        [
                            patheffects.withStroke(
                                linewidth=stroke_width, foreground=stroke
                            )
                        ]
                    )

        weights = None
        if bin_width_mode == "total":
            totals = mats[w].get("totals")
            if totals is not None:
                weights = np.asarray(totals, dtype=float)[perm]
        _draw_circle(
            ax_circle,
            M,
            ordered_labels,
            node_colors,
            cmap=cmap,
            vmin=0,
            vmax=vmax_used,
            weights=weights,
            rotate=rotate,
            fontsize_names=circle_name_font,
            label_box_pad=label_box_pad,
        )

        right_margin = 0.88
        bottom_margin = 0.22 if legend_groups and group_col_for_plot else 0.12
        fig.subplots_adjust(right=right_margin, bottom=bottom_margin, wspace=0.35)

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
            cb.set_label(label, fontsize=cb_label_font)
            cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            cb.ax.tick_params(labelsize=cb_tick_font)

        if legend_groups and group_col_for_plot:
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
                title=group_col_for_plot,
                loc="lower center",
                ncol=min(len(handles), 6),
                frameon=False,
                prop={"size": legend_font},
                title_fontsize=legend_title_font,
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

    # Circos path
    fig, axes = plt.subplots(
        1,
        len(wins),
        subplot_kw={"projection": "polar"},
        figsize=figsize_used,
        facecolor="white",
    )
    if len(wins) == 1:
        axes = [axes]
    axs = axes
    # Leave margin for the colorbar and optional group legend
    right_margin = 0.88
    bottom_margin = 0.18 if legend_groups and group_col_for_plot else 0.08
    fig.subplots_adjust(right=right_margin, bottom=bottom_margin)

    for i, w in enumerate(wins):
        M = np.array(mats[w]["matrix"], dtype=int)
        M = M[np.ix_(perm, perm)]
        if cap_weights is not None:
            M = np.minimum(M, cap_weights)

        weights = None
        if bin_width_mode == "total":
            totals = mats[w].get("totals")
            if totals is not None:
                weights = np.asarray(totals, dtype=float)[perm]
        _draw_circle(
            axs[i],
            M,
            ordered_labels,
            node_colors,
            cmap=cmap,
            vmin=0,
            vmax=vmax_used,
            weights=weights,
            rotate=rotate,
            fontsize_names=circle_name_font,
            label_box_pad=label_box_pad,
        )
        axs[i].set_title(
            w,
            fontsize=circle_title_font,
            loc="left",
            pad=20 * scale,
            color="black",
        )

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
        cb.set_label(label, fontsize=cb_label_font)
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cb.ax.tick_params(labelsize=cb_tick_font)

    # ---- groups legend
    if legend_groups and group_col_for_plot:
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
            title=group_col_for_plot,
            loc="lower center",
            ncol=min(len(handles), 6),
            frameon=False,
            prop={"size": legend_font},
            title_fontsize=legend_title_font,
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
