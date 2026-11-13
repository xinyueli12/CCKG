
import argparse
import json

import numpy as np
import pandas as pd

CSV_DEFAULT = "step8_risk_scores_v7.csv"
MIN_WINDOWS = 30        # matches the Step 8 caregiver table
MIN_CG = 5
TOP_Q = 0.90
SCORE = "risk_cohort"


def eta_squared(values, groups):
    v = pd.Series(values, dtype=float)
    g = pd.Series(groups).reset_index(drop=True)
    v = v.reset_index(drop=True)
    grand = v.mean()
    ss_tot = ((v - grand) ** 2).sum()
    ss_bet = v.groupby(g).apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    return float(ss_bet / ss_tot) if ss_tot > 0 else float("nan")


def profile(cg, by):
    top_cut = cg["mean_risk"].quantile(TOP_Q)
    cg = cg.assign(top=cg["mean_risk"] >= top_cut)
    n_cg, n_top = len(cg), int(cg["top"].sum())
    rows = []
    for key, g in cg.groupby(by):
        share_cg = len(g) / n_cg
        share_top = g["top"].sum() / n_top if n_top else np.nan
        share_top_wr = g["top_within_role"].sum() / cg["top_within_role"].sum()
        rows.append({
            by: key, "n_caregivers": len(g), "n_windows": int(g["n_windows"].sum()),
            "mean_of_cg_mean_risk": g["mean_risk"].mean(),
            "median_of_cg_mean_risk": g["mean_risk"].median(),
            "mean_demand": g["mean_demand"].mean(),
            "mean_recovery": g["mean_recovery"].mean(),
            "share_of_caregivers": share_cg,
            "share_of_top_decile": share_top,
            "representation_ratio": share_top / share_cg if share_cg else np.nan,
            "share_of_top_decile_within_role_rank": share_top_wr,
            "thin": len(g) < MIN_CG})
    return (pd.DataFrame(rows)
            .sort_values("n_caregivers", ascending=False).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    need = {"cgid", "role", "careUnit", SCORE, "demand_cohort", "recovery_cohort"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"FATAL: missing columns {sorted(miss)}; use the "
                         "window-level Step 8 v6/v7 export.")
    print("CCKG risk profile by role and unit (cohort-normalized score)\n" + "=" * 64)
    n_null_role = int(df["role"].isna().sum())
    df["role"] = df["role"].fillna("(missing)")

    # window level
    win_role = (df.groupby("role")[SCORE]
                .agg(n_windows="size", mean="mean", median="median",
                     p90=lambda s: s.quantile(0.90)))

    # caregiver level; a caregiver's unit is the unit of most of its windows
    cg = (df.groupby("cgid")
          .agg(role=("role", "first"),
               unit=("careUnit", lambda s: s.mode().iat[0]),
               n_windows=(SCORE, "size"), mean_risk=(SCORE, "mean"),
               mean_demand=("demand_cohort", "mean"),
               mean_recovery=("recovery_cohort", "mean"))
          .reset_index())
    roles_per_cg = df.groupby("cgid")["role"].nunique()
    cg = cg[cg["n_windows"] >= MIN_WINDOWS].copy()
    cg["pct_within_role"] = cg.groupby("role")["mean_risk"].rank(pct=True)
    cg["top_within_role"] = cg["pct_within_role"] >= TOP_Q

    eta_role = eta_squared(cg["mean_risk"], cg["role"])
    eta_unit = eta_squared(cg["mean_risk"], cg["unit"])
    eta_role_dem = eta_squared(cg["mean_demand"], cg["role"])
    eta_role_rec = eta_squared(cg["mean_recovery"], cg["role"])

    print(f"caregivers ranked (>= {MIN_WINDOWS} windows): {len(cg):,}; "
          f"windows with missing role: {n_null_role:,}; caregivers with more "
          f"than one role label: {int((roles_per_cg > 1).sum())}")
    print(f"\nvariance in caregiver mean risk explained by role: {eta_role:.1%}")
    print(f"                                       by unit: {eta_unit:.1%}")
    print(f"role explains {eta_role_dem:.1%} of variance in mean DEMAND and "
          f"{eta_role_rec:.1%} in mean RECOVERY")

    pr, pu = profile(cg, "role"), profile(cg, "unit")
    cols = ["n_caregivers", "mean_of_cg_mean_risk", "mean_demand", "mean_recovery",
            "share_of_caregivers", "share_of_top_decile", "representation_ratio",
            "share_of_top_decile_within_role_rank", "thin"]
    with pd.option_context("display.width", 200, "display.float_format",
                           "{:.3f}".format):
        print("\nBY ROLE (top decile = pooled ranking; last column = within-role rank)")
        print(pr[["role"] + cols].to_string(index=False))
        print("\nBY UNIT")
        print(pu[["unit"] + cols].to_string(index=False))
        print("\nWINDOW-LEVEL RISK BY ROLE")
        print(win_role.sort_values("n_windows", ascending=False).to_string())

    pr.to_csv("risk_profile_role.csv", index=False)
    pu.to_csv("risk_profile_unit.csv", index=False)
    json.dump({"input": args.csv, "score": SCORE, "min_windows": MIN_WINDOWS,
               "n_caregivers_ranked": int(len(cg)),
               "eta2_role": eta_role, "eta2_unit": eta_unit,
               "eta2_role_on_demand": eta_role_dem,
               "eta2_role_on_recovery": eta_role_rec,
               "windows_missing_role": n_null_role},
              open("risk_profile_results.json", "w"), indent=2)
    print("\nwrote risk_profile_role.csv, risk_profile_unit.csv, "
          "risk_profile_results.json")


if __name__ == "__main__":
    main()
