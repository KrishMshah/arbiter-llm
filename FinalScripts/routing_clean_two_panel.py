"""
FinalScripts/routing_clean_two_panel.py      (FREE, no model calls, runs in seconds)

Redraws the compact routing figure for the paper from the tables that routing_pgr_apgr_cpt.py
already wrote, using ONLY MT-Bench and SummEval.

Why: the GPT-5.6 pointwise run (item 10) was shown an out-of-fold arbitrator estimate that was made
with the two label-derived features. On TruthfulQA and FActScore (and therefore the "all 759 items"
panel) that estimate was too good, so those panels are not valid. On MT-Bench and SummEval the
arbitrator's error changes by under 0.05 when the two features are removed, so those two panels are
kept. The six-panel and 759-item figures from routing_pgr_apgr_cpt.py should not be used in the paper.

Reads : results/final/tables/routing_quality_vs_share_sent_to_strong_curves.csv
        results/final/tables/routing_weak_vs_strong_end_points_by_measure.csv
        results/final/tables/routing_cost_per_item_cascade_vs_always_escalate.csv
Writes: results/final/plots/routing_quality_vs_share_sent_to_gpt56_paper_two_panel.png
        (same file name as before, so it overwrites the old two-panel figure)

Run: python FinalScripts/routing_clean_two_panel.py
"""

import pandas as pd

from common import *

PANELS = [("MAE vs human score, MT-Bench (human)", "MT-Bench (195 items)"),
          ("MAE vs human score, SummEval", "SummEval (195 items)")]


def main():
    plt = setup_style()
    curves = pd.read_csv(TABLES / "routing_quality_vs_share_sent_to_strong_curves.csv")
    ends = pd.read_csv(TABLES / "routing_weak_vs_strong_end_points_by_measure.csv").set_index("measure")
    cost = pd.read_csv(TABLES / "routing_cost_per_item_cascade_vs_always_escalate.csv")
    deployed_share = float(cost["deployed_share_sent_to_terra_pct"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, (measure, title) in zip(axes, PANELS):
        c = curves[curves["measure"] == measure].sort_values("share_sent_to_strong_pct")
        assert len(c) > 0, f"measure not found in curves table: {measure}"
        ax.fill_between(c["share_sent_to_strong_pct"], c["router_ci_lo"], c["router_ci_hi"],
                        color=BLUE, alpha=0.13, lw=0)
        ax.plot(c["share_sent_to_strong_pct"], c["router"], "-o", color=BLUE, ms=2.5, lw=1.2, label="Router")
        ax.plot(c["share_sent_to_strong_pct"], c["random_router_expected"], "--", color=GREY, lw=1.1,
                label="Random router")
        ax.plot(deployed_share, ends.loc[measure, "router_at_deployed_rule"], "s", color=ORANGE, ms=5.5,
                zorder=5, label="Deployed rule")
        ax.set_xlabel("Items sent to GPT-5.6 (%)", fontsize=8.5)
        ax.set_ylabel("MAE vs human score", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    save_fig(fig, "routing_quality_vs_share_sent_to_gpt56_paper_two_panel.png")
    plt.close(fig)


if __name__ == "__main__":
    main()