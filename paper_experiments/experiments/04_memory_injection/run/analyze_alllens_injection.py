"""All-lens injection analysis: site selection quality + gain-vs-damage.

For every (model, dataset, tau, lens) with artifacts on disk:
  * lens pick  a_D = argmax_l [ mean P_lens(ans|h_exp[l]) - mean P_lens(ans|h_imp[l]) ]
    (two clean readout sweeps, tweak_1 - tweak_0; tau-independent)
  * scored against the lens-independent causal profile E_l (model's own final
    P(answer) after patching layer l), including the damage columns:
      causal_KL_tau*       KL(P_injected || P_clean) at the final position (nats)
      causal_top1keep_tau* fraction of examples whose argmax token is unchanged
  * gain captured g(l) = (E[l]-P_obs) / (max_l E[l]-P_obs)
  * efficiency = (E[l]-P_obs) / KL[l]   ("answer-probability gained per nat of
    collateral distortion") -- the targetedness measure.

Rows also include the depth heuristic (last layer) and the causal oracle
(argmax E) for reference. Output: alllens_injection_summary.csv + printed table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = {
    "GPT-2":       ("results_gpt2_expanded",    ["logit", "is_r64", "topk_r64", "tuned"]),
    "LLaMA-3-8B":  ("results_llama8b_expanded", ["logit", "is_r64", "topk_r64"]),
    "LLaMA-3-70B": ("results_llama70b_expanded", ["logit", "is_r64"]),
}
# --parity250: score the 8B picks from the parity250 checkpoints (identical
# 250-step annealed schedule; matches the paper's tables) and write to a
# separate CSV so the original summary is preserved. The causal profile is
# lens-independent and still read from is_r64.
PARITY250 = "--parity250" in sys.argv
if PARITY250:
    MODELS["LLaMA-3-8B"] = ("results_llama8b_expanded",
                            ["logit", "is_parity250", "topk_parity250", "tuned_parity250"])
DATASETS = ["hand", "2wmh"]


def lens_pick(d: Path, nL: int):
    t0 = pd.read_csv(d / "tweak_0.csv")
    t1 = pd.read_csv(d / "tweak_1.csv")
    D = np.array([t1[f"ans_prob_lens_edit_layer{l}"].mean()
                  - t0[f"ans_prob_lens_edit_layer{l}"].mean() for l in range(nL)])
    return int(D.argmax()), D, float(t0.answer_prob_obs.mean())


def main():
    rows = []
    for mname, (root, lenses) in MODELS.items():
        for ds in DATASETS:
            cpf = ROOT / root / "is_r64" / ds / "causal_profile.csv"
            if not cpf.exists():
                continue
            cp = pd.read_csv(cpf)
            nL = len(cp)
            taus = sorted(c.split("tau")[1] for c in cp.columns if c.startswith("causal_P_ans_tau"))
            has_damage = any(c.startswith("causal_KL_tau") for c in cp.columns)
            for tau in taus:
                E = cp[f"causal_P_ans_tau{tau}"].values
                KL = cp[f"causal_KL_tau{tau}"].values if has_damage else np.full(nL, np.nan)
                R = cp[f"causal_top1keep_tau{tau}"].values if has_damage else np.full(nL, np.nan)
                # P_obs from the head+IS tweak_0 (model-only quantity, lens-irrelevant)
                _, _, pobs = lens_pick(ROOT / root / "is_r64" / ds, nL)
                aE = int(E.argmax())
                den = E[aE] - pobs

                def emit(chooser, pick):
                    g = 100 * (E[pick] - pobs) / den if den > 0 else np.nan
                    eff = (E[pick] - pobs) / KL[pick] if KL[pick] and np.isfinite(KL[pick]) else np.nan
                    rows.append(dict(model=mname, ds=ds, tau=tau, chooser=chooser,
                                     pick=pick, causal_best=aE, nL=nL,
                                     gain_pct=g, P_at_pick=E[pick], P_obs=pobs,
                                     KL_at_pick=KL[pick], top1keep=R[pick], eff=eff,
                                     null=(den <= 0)))

                emit("oracle(argmaxE)", aE)
                emit("heuristic(last)", nL - 1)
                for lens in lenses:
                    d = ROOT / root / lens / ds
                    if not (d / "tweak_1.csv").exists():
                        continue
                    aD, _, _ = lens_pick(d, nL)
                    emit(f"lens:{lens}", aD)

    df = pd.DataFrame(rows)
    out = ROOT / ("alllens_injection_summary_parity250.csv" if PARITY250
                  else "alllens_injection_summary.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220, "display.max_rows", 400)
    fmt = lambda x: f"{x:9.4g}" if isinstance(x, float) else str(x)
    print(df.to_string(index=False, float_format=fmt))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
