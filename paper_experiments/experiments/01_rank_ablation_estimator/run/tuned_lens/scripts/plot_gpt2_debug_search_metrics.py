#!/usr/bin/env python3

import colorsys
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_ROOT = Path(
    "/path/to/data/tuned-lens/evaluation/gpt2_debug_search"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "plots"
LOGIT_BASELINE_PATH = Path(
    "/path/to/data/tuned-lens/evaluation/tl-gpt2-baseline/aggregate_metrics.json"
)
METRICS = ("kl", "ce", "entropy")
FAMILY_ORDER = ("logit", "tuned", "lora_kl", "topk", "head_is", "tail_only", "other")
FAMILY_BASE = {
    "logit": "#b91c1c",
    "tuned": "#222222",
    "lora_kl": "#7c3aed",
    "topk": "#2563eb",
    "head_is": "#059669",
    "tail_only": "#d97706",
    "other": "#6b7280",
}


def layer_key(key):
    return int(key[len("layer_"):])


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb01_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(1, rgb[0])) * 255),
        int(max(0, min(1, rgb[1])) * 255),
        int(max(0, min(1, rgb[2])) * 255),
    )


def adjust_lightness(hex_color, amount):
    r, g, b = hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.22, min(0.82, l + amount))
    return rgb01_to_hex(colorsys.hls_to_rgb(h, l, s))


def lens_family(name):
    if name == "logit_lens":
        return "logit"
    if name == "tuned_kl_baseline":
        return "tuned"
    if name.startswith("lora_kl_"):
        return "lora_kl"
    if "subset_topk" in name:
        return "topk"
    if "subset_head_is" in name:
        return "head_is"
    if "tail_only" in name:
        return "tail_only"
    return "other"


def lens_sort_key(lens):
    family = lens_family(lens["name"])
    return (FAMILY_ORDER.index(family), lens["name"])


def assign_family_colors(series):
    grouped = {}
    for lens in series:
        family = lens_family(lens["name"])
        grouped.setdefault(family, []).append(lens)

    for family, family_series in grouped.items():
        family_series.sort(key=lambda lens: lens["name"])
        base = FAMILY_BASE[family]
        count = len(family_series)
        for idx, lens in enumerate(family_series):
            if count == 1:
                lens["color"] = base
            else:
                offset = -0.14 + 0.28 * idx / (count - 1)
                lens["color"] = adjust_lightness(base, offset)
            lens["family"] = family
    return series


def load_metric_series(root, metric):
    series = []
    for metrics_path in sorted(root.glob("*/aggregate_metrics.json")):
        data = json.loads(metrics_path.read_text())
        tuned_metric = data["tuned"][metric]
        ordered = sorted(tuned_metric.items(), key=lambda item: layer_key(item[0]))
        series.append(
            {
                "name": metrics_path.parent.name,
                "layers": [layer_key(k) for k, _ in ordered],
                "values": [float(v) for _, v in ordered],
            }
        )

    if LOGIT_BASELINE_PATH.exists():
        data = json.loads(LOGIT_BASELINE_PATH.read_text())
        logit_metric = data["logit"][metric]
        ordered = sorted(logit_metric.items(), key=lambda item: layer_key(item[0]))
        series.append(
            {
                "name": "logit_lens",
                "layers": [layer_key(k) for k, _ in ordered],
                "values": [float(v) for _, v in ordered],
            }
        )

    if not series:
        raise ValueError("No aggregate_metrics.json files found under {}".format(root))
    series.sort(key=lens_sort_key)
    return assign_family_colors(series)


def render_png(metric, series, output_path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fffdf8")

    for lens in series:
        linestyle = "--" if lens["name"] == "logit_lens" else "-"
        linewidth = 2.8 if lens["name"] == "logit_lens" else 2.0
        markersize = 5.0 if lens["name"] == "logit_lens" else 4.5
        ax.plot(
            lens["layers"],
            lens["values"],
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            label=lens["name"],
            color=lens["color"],
            linestyle=linestyle,
        )

    ax.set_title("gpt2_debug_search tuned lens {}".format(metric.upper()), fontsize=20)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel(metric.upper(), fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(series[0]["layers"])

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    input_root = DEFAULT_INPUT_ROOT
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        series = load_metric_series(input_root, metric)
        render_png(metric, series, output_root / "{}_comparison.png".format(metric))

    print("Wrote plots to {}".format(output_root))


if __name__ == "__main__":
    main()
