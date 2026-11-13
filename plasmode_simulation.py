
import argparse
import json
import platform

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

CONFIG = {
    "scored_windows": "step8_risk_scores_v7.csv",
    "hsops_hospital": "hsops2_db2022_hospital.csv",
    "wps_hospital":   "wps_composites_hospital_level.csv",
    "seed": 20260802,
    "B": 500,
    "noise_levels": [0.25, 0.5, 1.0, 2.0],
    "lambda_demand": 0.30,
    "gamma_burnout": 0.60,
    "recovery_threshold": 0.95,
    "n_boot": 1000,
}
COLUMN_MAP = {"cgid": "CGID", "demand_cohort": "DemandCohort",
              "recovery_cohort": "RecoveryAdequacy", "risk_cohort": "Risk"}

# Table IV as published: (claim, noise) -> (median rho, sign rate, full rate)
PUBLISHED = {
    ("P1_staffing_demand", 0.25): (-0.53, 1.000, 0.93),
    ("P1_staffing_demand", 0.50): (-0.52, 1.000, 0.91),
    ("P1_staffing_demand", 1.00): (-0.52, 1.000, 0.91),
    ("P1_staffing_demand", 2.00): (-0.50, 0.998, 0.91),
    ("P2_risk_burnout",    0.25): (+0.97, 1.000, 1.00),
    ("P2_risk_burnout",    0.50): (+0.89, 1.000, 1.00),
    ("P2_risk_burnout",    1.00): (+0.70, 1.000, 0.998),
    ("P2_risk_burnout",    2.00): (+0.44, 0.994, 0.78),
}


def load_scored(path):
    win = pd.read_csv(path)
    if not {"CGID", "DemandCohort", "RecoveryAdequacy", "Risk"} <= set(win.columns):
        missing = set(COLUMN_MAP) - set(win.columns)
        assert not missing, f"{path} missing columns {sorted(missing)}"
        win = win.rename(columns=COLUMN_MAP)
    win = win[["CGID", "DemandCohort", "RecoveryAdequacy", "Risk"]]
    for c in ("DemandCohort", "RecoveryAdequacy"):
        assert win[c].between(0, 1).all(), f"{c} outside [0,1]"
    return win


def build_donor_pool(cfg):
    hs = pd.read_csv(cfg["hsops_hospital"])
    wps = pd.read_csv(cfg["wps_hospital"])
    d = wps[["PUBLICID", "burnout_pct_3plus"]].merge(
        hs[["PUBLICID", "POS_STAFF"]], on="PUBLICID", how="inner")
    assert len(d) == 38, f"Expected the 38-hospital joint set, got {len(d)}."
    # POS_STAFF is already a proportion. Do not divide by 100.
    d["staffing_adequacy"] = d["POS_STAFF"]
    assert d["staffing_adequacy"].between(0.05, 1).all(), (
        "staffing adequacy outside a plausible proportion range; check that "
        "POS_STAFF was not rescaled upstream")
    d["burnout_real"] = d["burnout_pct_3plus"] / 100.0
    assert d["burnout_real"].between(0, 1).all()
    print(f"  donor pool: {len(d)} hospitals; staffing adequacy "
          f"median {d.staffing_adequacy.median():.4f}, "
          f"range [{d.staffing_adequacy.min():.3f}, "
          f"{d.staffing_adequacy.max():.3f}]")
    return d[["PUBLICID", "staffing_adequacy", "burnout_real"]]


def boot_spearman_ci(x, y, n_boot, rng):
    """Percentile CI for Spearman rho by resampling the n hospitals."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    rx = rankdata(x[idx], axis=1)
    ry = rankdata(y[idx], axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx * ry).sum(axis=1) / denom
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan, np.nan
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def one_replication(win, donors, sigma_mult, cfg, rng):
    lam, gam = cfg["lambda_demand"], cfg["gamma_burnout"]
    cgs = win["CGID"].unique()
    pick = rng.integers(0, len(donors), size=len(cgs))
    assign = pd.DataFrame({
        "CGID": cgs,
        "PUBLICID": donors["PUBLICID"].to_numpy()[pick],
        "staffing_adequacy": donors["staffing_adequacy"].to_numpy()[pick]})
    d = win.merge(assign, on="CGID", how="left")

    deficit = d["staffing_adequacy"].mean() - d["staffing_adequacy"]
    eff1 = lam * deficit
    d["demand_sim"] = np.clip(
        d["DemandCohort"] + eff1
        + rng.normal(0.0, sigma_mult * eff1.std(), len(d)), 0.0, 1.0)
    d["risk_sim"] = (d["demand_sim"]
                     * (1.0 - d["staffing_adequacy"])
                     * (1.0 - d["RecoveryAdequacy"]))

    cg = d.groupby(["PUBLICID", "CGID"], as_index=False)[
        ["demand_sim", "risk_sim"]].mean()
    hosp = cg.groupby("PUBLICID", as_index=False)[
        ["demand_sim", "risk_sim"]].mean().merge(
        donors[["PUBLICID", "staffing_adequacy"]], on="PUBLICID")

    eff2 = gam * (hosp["risk_sim"] - hosp["risk_sim"].mean())
    hosp["burnout_sim"] = np.clip(
        0.30 + eff2 + rng.normal(0.0, sigma_mult * eff2.std(), len(hosp)),
        0.0, 1.0)

    out = {}
    for name, x, y, planted in [
            ("P1_staffing_demand", hosp["staffing_adequacy"],
             hosp["demand_sim"], -1),
            ("P2_risk_burnout", hosp["risk_sim"], hosp["burnout_sim"], +1)]:
        rho = spearmanr(x, y).statistic
        lo, hi = boot_spearman_ci(x, y, cfg["n_boot"], rng)
        out[name] = {"rho": float(rho),
                     "sign_ok": bool(np.sign(rho) == planted),
                     "ci_excl0": bool(np.isfinite(lo) and ((lo > 0) or (hi < 0)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=CONFIG["scored_windows"])
    ap.add_argument("--B", type=int, default=CONFIG["B"])
    a = ap.parse_args()
    cfg = dict(CONFIG, scored_windows=a.scored, B=a.B)
    rng = np.random.default_rng(cfg["seed"])

    print("CCKG plasmode verification (v3)\n" + "=" * 62)
    win = load_scored(cfg["scored_windows"])
    print(f"  {len(win):,} scored windows, {win.CGID.nunique():,} caregivers")
    donors = build_donor_pool(cfg)

    rows = []
    for sig in cfg["noise_levels"]:
        tally = {k: {"sign": 0, "full": 0, "rhos": []}
                 for k in ("P1_staffing_demand", "P2_risk_burnout")}
        for b in range(cfg["B"]):
            r = one_replication(win, donors, sig, cfg, rng)
            for k, v in r.items():
                tally[k]["rhos"].append(v["rho"])
                tally[k]["sign"] += int(v["sign_ok"])
                tally[k]["full"] += int(v["sign_ok"] and v["ci_excl0"])
            if (b + 1) % 100 == 0:
                print(f"    noise {sig:<5} {b + 1}/{cfg['B']} replications",
                      flush=True)
        for k, t in tally.items():
            rows.append({"claim": k, "noise_mult": sig,
                         "median_rho": float(np.median(t["rhos"])),
                         "sign_rate": t["sign"] / cfg["B"],
                         "full_rate": t["full"] / cfg["B"],
                         "recovered": t["full"] / cfg["B"]
                         >= cfg["recovery_threshold"]})

    res = pd.DataFrame(rows)
    res.to_csv("plasmode_results.csv", index=False)
    print("\n" + res.to_string(index=False))

    label = {"P1_staffing_demand": r"Staffing$\to$Demand ($-$)",
             "P2_risk_burnout": r"Risk$\to$Burnout ($+$)"}
    with open("plasmode_table.tex", "w") as f:
        f.write(f"% plasmode_simulation_v3.py, seed {cfg['seed']}, "
                f"B={cfg['B']}, {cfg['n_boot']} bootstrap draws\n")
        f.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n"
                "Claim & Noise & med.\\ $\\rho$ & Sign & Full \\\\\n\\midrule\n")
        for i, claim in enumerate(label):
            if i:
                f.write("\\midrule\n")
            sub = res[res.claim == claim].sort_values("noise_mult")
            for j, (_, r) in enumerate(sub.iterrows()):
                name = label[claim] if j == 0 else ""
                f.write(f"{name} & {r.noise_mult:.2f} & {r.median_rho:+.2f} & "
                        f"{r.sign_rate:.1%} & {r.full_rate:.1%} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # verification gate against the published Table IV (default B only)
    if cfg["B"] == CONFIG["B"]:
        bad = []
        for r in rows:
            pub = PUBLISHED[(r["claim"], r["noise_mult"])]
            got = (round(r["median_rho"], 2), r["sign_rate"], r["full_rate"])
            if (got[0] != pub[0] or abs(got[1] - pub[1]) > 0.0015
                    or abs(got[2] - pub[2]) > 0.0051):
                bad.append((r["claim"], r["noise_mult"], got, pub))
        print("\nTable IV reproduction: "
              + ("PASS" if not bad else f"MISMATCH in {len(bad)} cells"))
        for b in bad:
            print(f"  {b[0]} noise {b[1]}: got {b[2]}, published {b[3]}")

    json.dump({"config": cfg, "rows": rows,
               "environment": {"python": platform.python_version(),
                               "numpy": np.__version__,
                               "pandas": pd.__version__}},
              open("plasmode_results.json", "w"), indent=2, default=float)
    print("\nwrote plasmode_results.csv, plasmode_table.tex, "
          "plasmode_results.json")


if __name__ == "__main__":
    main()
