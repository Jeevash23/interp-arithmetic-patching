#!/usr/bin/env python3
"""Figures of the manuscript, drawn from saved outputs only (make figures).

Reads the recounts in generated/reconciliation/numbers.json and the post hoc results in
../analysis/posthoc/results/, and Study 5's results/summary.json, and writes figures/fig-one-token.pdf,
fig-operand.pdf, fig-screen.pdf and fig-study5.pdf.
Needs matplotlib; nothing here runs a model. PDFs are written without a creation date, so reruns are byte-stable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
os.environ.setdefault("MPLCONFIGDIR", str(PAPER / "build/mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(PROJECT / "analysis/posthoc"))
from common import exact_interval  # noqa: E402

POSTHOC = PROJECT / "analysis/posthoc/results"
RECON = PAPER / "generated/reconciliation/numbers.json"
OUT = PAPER / "figures"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"   # validated categorical order (light surface)
META = {"CreationDate": None}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
    "hatch.linewidth": 0.6,
})


def load(path):
    return json.loads(Path(path).read_text())


def share(k, n):
    lo, hi = exact_interval(k, n)
    p = k / n
    return p, (p - lo, hi - p)


def bars(ax, xs, counts, color, width, hatch=None, face=None):
    ps = [share(k, n) for k, n in counts]
    ax.bar(xs, [p for p, _ in ps], width=width, color=face or color, edgecolor=color if hatch else "none",
           hatch=hatch, lw=0.6 if hatch else 0, zorder=3)
    ax.errorbar(xs, [p for p, _ in ps], yerr=[[e[0] for _, e in ps], [e[1] for _, e in ps]], fmt="none",
                ecolor=INK, elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=4)
    for x, (p, e), (k, n) in zip(xs, ps, counts):
        ax.text(x, p + e[1] + 0.025, f"{k}/{n}", ha="center", va="bottom", fontsize=5.8, color=INK2, rotation=90)


def fig_one_token():
    R = {k: v["value"] for k, v in load(RECON).items()}
    tr, fde = load(POSTHOC / "transfer-rates.json"), load(POSTHOC / "first-digit-equivalence.json")
    ro, s2 = tr["readout_qwen3_06b"], tr["study2_qwen3_06b"]
    f23 = R["ai02.fresh.Block 23, answer prefix"][0]
    qsel = R["ai02.qualification.Qwen3-4B-FP8, instruction plus three demonstrations"]
    kn = lambda r: (r["k"], r["n"])  # noqa: E731
    groups = ["Readout, 0.6B\nblock 26, labelled", "Held-out, 0.6B\nblock 26, first literal",
              "Held-out, 0.6B\nblock 26, labelled", "Answer-only, 4B\nblock 23, answer prefix"]
    act = [kn(ro["activation"]), kn(s2["controls/first_z-L26/activation"]), kn(s2["controls/label_z-L26/activation"]),
           (f23["donor"]["counterfactual"], f23["donor"]["n"])]
    digit = [kn(ro["text_first_digit"]), kn(s2["controls/first_z-L26/text_digit"]),
             kn(s2["controls/label_z-L26/text_digit"]),
             (f23["text_first_answer_digit"]["counterfactual"], f23["text_first_answer_digit"]["n"])]
    value = [kn(ro["text_full_value"]), kn(s2["controls/first_z-L26/text_full"]), kn(s2["controls/label_z-L26/text_full"])]
    baseline = [tuple(qsel["textual_counterfactual"])]
    dp = R["ai04.controls.digit_pairs"]
    b23 = next(s for s in fde["qwen3_4b"] if s["write"] == "whole_vector" and s["site"] == "Block 23, answer prefix")
    equiv = [tuple(R["ai02.readout"]["first_digit_exact_sequence"]),
             (dp["first_z-L26"]["exact_sequence"], dp["first_z-L26"]["runnable_pairs"]),
             (dp["label_z-L26"]["exact_sequence"], dp["label_z-L26"]["runnable_pairs"]),
             (b23["identical_to_text_edit"], b23["pairs"])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.1), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    w = 0.22
    x = list(range(4))
    bars(ax1, [i - w for i in x], act, BLUE, w - 0.03)
    bars(ax1, x, digit, ORANGE, w - 0.03)
    bars(ax1, [i + w for i in x[:3]], value, AQUA, w - 0.03)
    bars(ax1, [3 + w], baseline, AQUA, w - 0.03, hatch="//////", face=SURFACE)
    ax1.set_ylabel("Complete counterfactual\nanswers")
    ax1.set_title("(a) Transfer of the donor's intermediate", color=INK, pad=3, loc="left")
    handles = [Patch(color=BLUE, label="Activation write"), Patch(color=ORANGE, label="First digit as text"),
               Patch(color=AQUA, label="Value as text"),
               Patch(facecolor=SURFACE, edgecolor=AQUA, hatch="//////", lw=0.6, label="Prompt baseline (4B)")]
    ax1.legend(handles=handles, loc="upper left", ncol=2, fontsize=7, handlelength=1.4, columnspacing=1.0)
    bars(ax2, x, equiv, BLUE, 0.3)
    ax2.set_ylabel("Activation output equal\nto first-digit text output")
    ax2.set_title("(b) Activation output compared with the first-digit text edit, token for token", color=INK, pad=3,
                  loc="left")
    ax2.set_xticks(x, groups)
    ax2.tick_params(axis="x", length=0, labelsize=7.2)
    for ax in (ax1, ax2):
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax1.set_xlim(-0.55, 3.55)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(OUT / "fig-one-token.pdf", metadata=META)
    plt.close(fig)


SITE_LABELS = {"Block 20, first operand": "Block 20\nfirst operand", "Block 9, first operand": "Block 9\nfirst operand",
               "Block 10, first operand": "Block 10\nfirst operand", "Block 14, second operand": "Block 14\nsecond operand",
               "Block 20, first-stage end": "Block 20\nend of first step",
               "Block 23, answer prefix": "Block 23\nanswer prefix"}
OTHERS = ["same_z_decomposition", "same_z_different_answer", "different_z_same_answer", "same_leading_different_value",
          "random_source"]


def fig_operand():
    sites = load(POSTHOC / "operand-substitution.json")["sites"]
    order = [s["site"] for s in sites]
    assert order == list(SITE_LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 2.45))
    offsets = [-0.24, -0.12, 0.0, 0.12, 0.24]
    for i, s in enumerate(sites):
        for dx, cond in zip(offsets, OTHERS):
            r = s["sources"][cond]
            p, e = share(r["k"], r["n"])
            ax.errorbar([i + 0.08 + dx * 0.6], [p], yerr=[[e[0]], [e[1]]], fmt="o", ms=3.6, color=MUTED, mec=SURFACE,
                        mew=0.6, elinewidth=0.5, capsize=0, zorder=3)
        r = s["sources"]["donor"]
        p, e = share(r["k"], r["n"])
        ax.errorbar([i - 0.22], [p], yerr=[[e[0]], [e[1]]], fmt="o", ms=5.5, color=BLUE, mec=SURFACE, mew=1.0,
                    elinewidth=0.7, capsize=1.6, capthick=0.6, zorder=5)
        ax.text(i - 0.30, p, f"{r['k']}/{r['n']}", ha="right", va="center", fontsize=6.2, color=INK2)
        r = s["controls"]["isotropic_delta"]
        ax.plot([i + 0.38], [r["k"] / r["n"]], marker="x", ms=4.5, mew=0.9, color=INK, ls="none", zorder=4)
    ax.set_xticks(range(len(sites)), [SITE_LABELS[s] for s in order])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.75, len(sites) - 0.4)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Answers equal to the program\nrun on the source's operand")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="none", ms=5.5, color=BLUE, mec=SURFACE, label="Donor"),
               Line2D([], [], marker="o", ls="none", ms=3.6, color=MUTED, mec=SURFACE, label="Five other sources"),
               Line2D([], [], marker="x", ls="none", ms=4.5, mew=0.9, color=INK, label="Matched noise")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig-operand.pdf", metadata=META)
    plt.close(fig)


def fig_screen():
    scr = load(POSTHOC / "screen-decomposition.json")["sites"]
    tr = load(POSTHOC / "transfer-rates.json")["qwen3_4b"]
    tested = {(20, "a_end"): "Block 20, first operand", (9, "a_end"): "Block 9, first operand",
              (10, "a_end"): "Block 10, first operand", (14, "b_end"): "Block 14, second operand",
              (20, "first_stage_end"): "Block 20, first-stage end", (23, "prompt_end"): "Block 23, answer prefix"}
    roles = [("a_end", "First operand"), ("b_end", "Second operand"), ("first_stage_end", "End of first step"),
             ("prompt_end", "Answer prefix")]
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.7), sharex=True, sharey="row")
    for col, (role, title) in enumerate(roles):
        rows = sorted((s for s in scr if s["role"] == role), key=lambda s: s["layer"])
        L = [s["layer"] for s in rows]
        top, bottom = axes[0, col], axes[1, col]
        top.plot(L, [s["delta_log_ratio"] for s in rows], color=BLUE, lw=1.4, label="Screen statistic", zorder=3)
        top.plot(L, [s["recipient_suppression"] for s in rows], color=ORANGE, lw=1.1,
                 label="Suppression of the recipient's answer", zorder=3)
        top.plot(L, [s["target_gain"] for s in rows], color=AQUA, lw=1.1, label="Gain of the target", zorder=3)
        bottom.plot(L, [s["mean_p_target"] for s in rows], color=BLUE, lw=1.4, zorder=3)
        bottom.set_yscale("log")
        for s in rows:
            key = (s["layer"], role)
            if key not in tested:
                continue
            r = tr["whole_vector/" + tested[key]]
            for ax, val in ((top, s["delta_log_ratio"]), (bottom, s["mean_p_target"])):
                ax.plot([s["layer"]], [val], marker="o", ms=5, mfc=SURFACE, mec=INK, mew=0.9, ls="none", zorder=5)
            dx = {9: -9, 10: 9}.get(s["layer"], 0) if role == "a_end" else 0
            bottom.annotate(f"{r['k']}/{r['n']}", (s["layer"], s["mean_p_target"]), textcoords="offset points",
                            xytext=(dx, 7), ha="center", fontsize=6.2, color=INK2)
        top.set_title(title, color=INK, pad=3)
        for ax in (top, bottom):
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
        bottom.set_xlabel("Block")
        bottom.set_xticks([0, 10, 20, 30])
    axes[0, 0].set_ylabel("Mean change (nats)")
    axes[1, 0].set_ylabel("Mean P(target)")
    axes[1, 0].set_ylim(1e-10, 2)
    axes[1, 0].set_yticks([1e-9, 1e-6, 1e-3, 1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="none", ms=5, mfc=SURFACE, mec=INK, mew=0.9))
    labels.append("Site tested with generation (complete transfers / pairs)")
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=0.6, w_pad=0.4)
    fig.savefig(OUT / "fig-screen.pdf", metadata=META)
    plt.close(fig)


def fig_study5():
    """Study 5 by block: the added direction (alpha = 1) at the end of the first step and at the answer prefix and the two
    span writes on the held-out pairs; the teacher-forced score against generated answers at the answer prefix, and the
    direction at the first operand, on the development pairs (the cost rule dropped the operand positions from the test)."""
    s5 = load(PROJECT / "studies/5-direction-and-span/results/summary.json")
    test, dev = s5["test"]["sweeps"], s5["development"]["sweeps"]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0), sharex=True, sharey=True)

    def lines(ax, rows, first_op=False, zsplice=False):
        L = [r["layer"] for r in rows]
        ax.plot(L, [1 - r["unchanged"] / r["n"] for r in rows], color=MUTED, lw=1.1, zorder=3)
        ax.plot(L, [r["cf"] / r["n"] for r in rows], color=BLUE, lw=1.4, zorder=5)
        if first_op:
            ax.plot(L, [r["first_op"] / r["n"] for r in rows], color=ORANGE, lw=1.1, ls="--", zorder=4)
        if zsplice:
            ax.plot(L, [r["z_splice"] / r["n"] for r in rows], color=ORANGE, lw=1.1, zorder=4)

    panels = [
        (axes[0, 0], "(a) Direction, end of first step", lambda ax: lines(ax, test["direction"]["first_stage_end/1"], zsplice=True)),
        (axes[0, 1], "(b) Direction, answer prefix", lambda ax: lines(ax, test["direction"]["prompt_end/1"], zsplice=True)),
        (axes[1, 0], "(d) Span after the second operand", lambda ax: lines(ax, test["span"]["post_operand"], first_op=True)),
        (axes[1, 1], "(e) Span of the whole first step", lambda ax: lines(ax, test["span"]["first_step"], first_op=True)),
        (axes[1, 2], "(f) Direction, first operand (dev.)", lambda ax: lines(ax, dev["direction"]["a_end/1"], zsplice=True)),
    ]
    for ax, title, draw in panels:
        draw(ax)
        ax.set_title(title, color=INK, pad=3, fontsize=7.3, loc="left")
    ax = axes[0, 2]
    rows = dev["scores"]["prompt_end"]
    gen = {r["layer"]: r for r in dev["direction"]["prompt_end/1"]}
    L = [r["layer"] for r in rows]
    ax.plot(L, [r["flips"] / r["n"] for r in rows], color=AQUA, lw=1.4, zorder=4)
    ax.plot(L, [gen[l]["cf"] / gen[l]["n"] for l in L], color=BLUE, lw=1.4, zorder=5)
    ax.set_title("(c) Answer prefix, scored (dev.)", color=INK, pad=3, fontsize=7.3, loc="left")
    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Block")
        ax.set_xticks([0, 10, 20, 30])
    axes[0, 0].set_ylabel("Share of pairs")
    axes[1, 0].set_ylabel("Share of pairs")
    handles = [Line2D([], [], color=MUTED, lw=1.1, label="Answer changed"),
               Line2D([], [], color=BLUE, lw=1.4, label="Complete counterfactual"),
               Line2D([], [], color=ORANGE, lw=1.1, label="Donor's leading digit in the intermediate"),
               Line2D([], [], color=ORANGE, lw=1.1, ls="--", label="Donor's first operation"),
               Line2D([], [], color=AQUA, lw=1.4, label="Counterfactual more likely (teacher-forced)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=0.8, w_pad=0.4)
    fig.savefig(OUT / "fig-study5.pdf", metadata=META)
    plt.close(fig)


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the figures to this folder instead of paper/figures (used by check_paper.py)")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(exist_ok=True)
    fig_one_token()
    fig_operand()
    fig_screen()
    fig_study5()
    print(f"wrote fig-one-token.pdf, fig-operand.pdf, fig-screen.pdf and fig-study5.pdf to {OUT}")


if __name__ == "__main__":
    main()
