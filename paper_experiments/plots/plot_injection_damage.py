"""fig_injection_damage_eff / fig_injection_damage_top1: gain per nat of KL
damage and top-1 preservation of each lens's pick. No per-bar labels
(adjacent bars share values); exact numbers live in tab:inject_damage."""
from case_common import CHOOSERS, S, grouped, load_summary
import matplotlib.pyplot as plt


def main():
    df = load_summary()
    df = df.assign(tau=df.tau.astype(str))
    ch = [c for c in CHOOSERS if c[0] != "lens:logit"]
    specs = [("eff", "$100\\cdot\\Delta P(\\mathrm{ans})$ / KL nat",
              "fig_injection_damage_eff.pdf"),
             ("top1keep", "top-1 token kept (%)",
              "fig_injection_damage_top1.pdf")]
    for col, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=S.SIZE_1COL, constrained_layout=True)
        grouped(ax, df, col, scale=100, choosers=ch, labels=False)
        ax.set_ylabel(ylabel)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=S.FONT_ANNOT)
        S.finalize(fig, S.FIGDIR / fname, sync=True)
        print(f"wrote {fname}")


if __name__ == "__main__":
    main()
