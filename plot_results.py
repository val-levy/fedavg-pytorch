#!/usr/bin/env python3
"""
Aggregate every results/*/lr_sweep_*.json into paper-style figures.

Produces (in --output, default results/figures):
  1. rounds_to_target.png  -- Table 2 style: rounds to reach the target
     accuracy per config, grouped by distribution (the headline FedAvg result).
  2. accuracy_vs_rounds_*.png -- Figure 2/7 style: test accuracy vs communication
     round for the best eta of each config. Only configs whose sweep JSON stored
     per-round "checkpoints" appear here (re-run sweeps after this update to populate).

Usage:  python plot_results.py [--output DIR] [--glob 'results/t2_*']
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_tag(tag):
    """Pull (table, dist, E, B, C, u) out of a result-dir tag like t2_iid_E20_B10
    (an optional model prefix such as cnn_ is tolerated)."""
    table = "Table 1" if "t1_" in tag else "Table 2"
    dist = "non-IID" if ("_ni_" in tag or "_ni2_" in tag) else "IID"
    E = int(m.group(1)) if (m := re.search(r"E(\d+)", tag)) else 1
    Bm = re.search(r"B(\w+?)(?:_|$)", tag)
    Braw = Bm.group(1) if Bm else "10"
    B = float("inf") if Braw == "inf" else int(Braw)
    Cm = re.search(r"C(\d+)", tag)
    C = {"00": 0.0, "02": 0.2, "05": 0.5, "10": 1.0, "01": 0.1}.get(Cm.group(1), 0.1) if Cm else 0.1
    u = E if B == float("inf") else 600 * E / B   # expected local updates/round (n/K=600)
    Blabel = "∞" if B == float("inf") else str(B)
    return {"table": table, "dist": dist, "E": E, "B": Blabel, "C": C, "u": u,
            "label": f"E={E} B={Blabel}" + ("" if table == "Table 2" else f" C={C}")}


def load(pattern):
    rows = []
    for f in glob.glob(pattern):
        tag = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        res = d.get("results", {})
        if not res:
            continue
        label = next(iter(res))
        info = res[label]
        best = info["best_lr"]
        bestentry = next(s for s in info["sweep"] if s["lr"] == best)
        meta = parse_tag(tag)
        rows.append({**meta, "tag": tag, "best_lr": best,
                     "target_round": bestentry["target_round"],
                     "final_acc": bestentry["final_acc"],
                     "checkpoints": bestentry.get("checkpoints")})
    return rows


def fig_rounds_to_target(rows, target, outpath):
    dists = ["IID", "non-IID"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for ax, dist in zip(axes, dists):
        sub = sorted([r for r in rows if r["dist"] == dist], key=lambda r: r["u"])
        if not sub:
            ax.set_visible(False)
            continue
        labels = [r["label"] for r in sub]
        vals = [r["target_round"] if r["target_round"] is not None else 0 for r in sub]
        colors = ["#4C78A8" if r["target_round"] is not None else "#BBBBBB" for r in sub]
        bars = ax.bar(range(len(sub)), vals, color=colors)
        for i, r in enumerate(sub):
            txt = str(r["target_round"]) if r["target_round"] is not None else "n/r"
            ax.text(i, vals[i], txt, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{dist}  (ordered by local updates/round u →)")
        ax.set_ylabel(f"Rounds to {target:.0%} accuracy")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("FedAvg MNIST 2NN: communication rounds to target accuracy (lower = better)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


# paper-style encoding: colour by batch size B, line style by local epochs E
B_COLORS = {"10": "#D62728", "50": "#F0A030", "∞": "#1F77B4"}   # red / orange / blue
E_STYLES = {1: "-", 5: "--", 20: ":"}


def fig_curves(rows, target, model_name, outpath, xmax=1000):
    """Two-panel accuracy-vs-rounds figure (IID | non-IID), Figure-2 style."""
    have = [r for r in rows if r["checkpoints"]]
    if not have:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    accs = []
    for ax, dist in zip(axes, ["IID", "non-IID"]):
        sub = sorted([r for r in have if r["dist"] == dist], key=lambda r: (r["B"], r["E"]))
        for r in sub:
            pts = sorted((int(k), v) for k, v in r["checkpoints"].items() if int(k) <= xmax)
            if not pts:
                continue
            xs, ys = zip(*pts)
            accs.extend(ys)
            ax.plot(xs, ys,
                    color=B_COLORS.get(r["B"], "#444"),
                    linestyle=E_STYLES.get(r["E"], "-"),
                    lw=1.6,
                    label=f"B={r['B']} E={r['E']}")
        ax.axhline(target, color="gray", lw=0.8)
        ax.set_title(f"MNIST {model_name} {dist}")
        ax.set_xlabel("Communication Rounds")
        ax.set_xlim(0, xmax)
        ax.grid(True, alpha=0.25)
        # order legend by B then E to mirror the paper
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            order = sorted(range(len(labels)), key=lambda i: labels[i])
            ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=7, ncol=1)
    axes[0].set_ylabel("Test Accuracy")
    lo = max(0.90, (min(accs) if accs else target) - 0.005)
    axes[0].set_ylim(lo, 1.0)
    fig.suptitle(f"FedAvg MNIST {model_name}: test accuracy vs. communication rounds")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/*/lr_sweep_*.json", help="glob for sweep result files")
    ap.add_argument("--output", default="results/figures", help="output dir for figures")
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--model-name", default="2NN", help="Model label for figure titles (e.g. CNN, 2NN)")
    args = ap.parse_args()

    rows = load(args.glob)
    if not rows:
        print(f"No sweep results matched {args.glob}")
        return
    os.makedirs(args.output, exist_ok=True)

    made = [fig_rounds_to_target(rows, args.target, os.path.join(args.output, "rounds_to_target.png"))]
    p = fig_curves(rows, args.target, args.model_name, os.path.join(args.output, "accuracy_vs_rounds.png"))
    if p:
        made.append(p)

    n_curves = sum(1 for r in rows if r["checkpoints"])
    print(f"Loaded {len(rows)} configs ({n_curves} with per-round curves).")
    for p in made:
        print("  wrote", p)
    if n_curves == 0:
        print("\nNote: no per-round curves found — re-run sweeps after the experiment.py "
              "update so 'checkpoints' get saved, then the accuracy-vs-rounds figures will populate.")


if __name__ == "__main__":
    main()
