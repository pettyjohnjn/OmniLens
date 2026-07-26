"""fig_site_selection_bars: causal gain captured by each lens's injection-site
pick, 8B/70B stack. Data: alllens_injection_summary_parity250.csv."""
from case_common import CHOOSERS, S, grouped, load_summary, tag
import matplotlib.pyplot as plt


def main():
    df = load_summary()
    df = df.assign(tau=df.tau.astype(str))
    # full-rank reference rows (tuned_parity250) exist only at 8B, so the
    # 70B panel simply omits that bar
    choosers = CHOOSERS[:2] + [(("lens:tuned_parity250",), "full-rank ref.", S.TUNED_RED, None)] \
        + CHOOSERS[2:]
    fig, axes = plt.subplots(2, 1, figsize=S.SIZE_1COL_STACK, constrained_layout=True)
    for ax, model in zip(axes, ["LLaMA-3-8B", "LLaMA-3-70B"]):
        groups = [(model, "2.0"), (model, "4.0")]
        grouped(ax, df, "gain_pct", choosers=choosers,
                groups=groups, group_labels=["$\\tau{=}2$", "$\\tau{=}4$"])
        ax.set_ylim(0, 124)
        ax.set_ylabel("causal gain captured (%)")
        tag(ax, model)
    # legend handles from the 8B panel: the 70B panel lacks some bars, and
    # empty bar containers render default-blue legend swatches
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.22), fontsize=S.FONT_ANNOT)
    S.finalize(fig, S.FIGDIR / "fig_site_selection_bars.pdf", sync=True)
    print("wrote fig_site_selection_bars")


if __name__ == "__main__":
    main()
