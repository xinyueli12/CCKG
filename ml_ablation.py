
import argparse
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import sklearn
import xgboost as xgb

CSV_DEFAULT = "step8_risk_scores.csv"
MIN_WINDOWS = 10
TRAIN_FRAC = 0.70
N_BOOT = 1000
SEED = 0
HIGH_LOAD_Q = 0.75                  # matches populate_mimic_step2.py
PUBLISHED = {"A": 134.300, "B": 89.600}
GATE_TOL = 0.01

CORE = ["EventCount_t", "DocumentationIntensity_t", "window_night_t"]
ONTO = ["InterWindowGap_t", "ConsecutiveHighLoadCount_t", "role_enc", "unit_enc"]
CATEGORICAL = {"role_enc", "unit_enc", "window_night_t", "first_window_t"}
PREREGISTERED = {"InterWindowGap_t": "negative",
                 "ConsecutiveHighLoadCount_t": "positive"}
XGB_PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                  n_jobs=4, objective="reg:squarederror")
LC_FRACS = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


# ------------------------------------------------------------------ panel
def build_panel(df):
    """Feature row at window t, target at window t+1 within the same stay.
    Identical to the submitted specification (ml_ablation / ml_robustness),
    so the verification gate can reproduce the published figures."""
    need = {"cgid", "stayid", "wstart", "wtype", "docInt", "evCount", "gap",
            "role", "careUnit"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"FATAL: CSV missing columns {sorted(missing)}. Use the "
                 "window-level Step 8 v6/v7 export.")
    streak_col = next((c for c in ("ConsecutiveHighLoadCount",
                                   "consecutiveHighLoad") if c in df.columns),
                      None)
    if streak_col is None:
        sys.exit("FATAL: streak column absent; the gate cannot reproduce the "
                 "submitted Condition B without it.")

    df = df.sort_values(["cgid", "stayid", "wstart"]).copy()
    g = df.groupby(["cgid", "stayid"], sort=False)
    out = pd.DataFrame({
        "cgid": df["cgid"].to_numpy(),
        "stayid": df["stayid"].to_numpy(),
        "wstart": df["wstart"].to_numpy(),
        "EventCount_t": df["evCount"].to_numpy(float),
        "DocumentationIntensity_t": df["docInt"].to_numpy(float),
        "window_night_t": (df["wtype"] == "night").astype(int).to_numpy(),
        "InterWindowGap_t": df["gap"].to_numpy(float),
        "streak_upstream": df[streak_col].to_numpy(float),
        "role": df["role"].to_numpy(),
        "unit": df["careUnit"].to_numpy(),
        "y_next": g["docInt"].shift(-1).to_numpy(),
    })
    out = out.dropna(subset=["y_next"]).reset_index(drop=True)
    out["first_window_t"] = out["InterWindowGap_t"].isna().astype(int)
    out["InterWindowGap_t"] = out["InterWindowGap_t"].fillna(0.0)
    counts = out.groupby("cgid").size()
    kept = out[out.cgid.isin(counts[counts >= MIN_WINDOWS].index)].copy()
    return kept, out


def temporal_split(panel, frac=TRAIN_FRAC):
    """Submitted split: first `frac` of each caregiver's rows, ordered by stay
    then window start. Temporal within a stay; not across stays."""
    tr, te = [], []
    for _, g in panel.groupby("cgid", sort=False):
        k = int(np.floor(len(g) * frac))
        if k < 1 or k >= len(g):
            continue
        tr.extend(g.index[:k])
        te.extend(g.index[k:])
    return panel.loc[tr], panel.loc[te]


def stay_grouped_split(panel, frac=TRAIN_FRAC, seed=SEED):
    """Sensitivity: every stay lies wholly in train or wholly in test.
    Within each caregiver, stays are taken in a seeded random order until at
    least `frac` of the caregiver's rows are in train; at least one stay is
    kept on each side. Caregivers with a single stay cannot be split and are
    reported, not silently dropped."""
    rng = np.random.default_rng(seed)
    tr, te, single = [], [], 0
    for _, g in panel.groupby("cgid", sort=False):
        stays = g["stayid"].unique()
        if len(stays) < 2:
            single += 1
            continue
        stays = stays[rng.permutation(len(stays))]
        sizes = g.groupby("stayid").size()
        cum, chosen = 0, []
        for s in stays[:-1]:                 # last stay always goes to test
            chosen.append(s)
            cum += sizes[s]
            if cum >= frac * len(g):
                break
        mask = g["stayid"].isin(chosen)
        tr.extend(g.index[mask])
        te.extend(g.index[~mask])
    return panel.loc[tr], panel.loc[te], single


# ----------------------------------------------------------------- features
def encode(train, test):
    """Ordinal codes for role/unit, fitted on TRAINING rows only."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr, te = train.copy(), test.copy()
    tr[["role_enc", "unit_enc"]] = enc.fit_transform(tr[["role", "unit"]])
    te[["role_enc", "unit_enc"]] = enc.transform(te[["role", "unit"]])
    return tr, te


def encode_legacy(panel_all, train, test):
    """Submitted behavior: encoder fitted on the full panel, before the
    MIN_WINDOWS filter. Used by the verification gate only."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    codes = pd.DataFrame(enc.fit_transform(panel_all[["role", "unit"]]),
                         index=panel_all.index, columns=["role_enc", "unit_enc"])
    tr, te = train.copy(), test.copy()
    tr[["role_enc", "unit_enc"]] = codes.loc[tr.index].to_numpy()
    te[["role_enc", "unit_enc"]] = codes.loc[te.index].to_numpy()
    return tr, te


def streak_from_threshold(rows, threshold_source):
    """Consecutive-high-load run length within each (caregiver, stay)
    sequence, with each caregiver's threshold fitted on `threshold_source`
    rows only. A caregiver absent from the source falls back to the pooled
    quantile of the source rows, which never includes test rows when the
    source is the training set. Returns (streak Series aligned to rows,
    number of caregivers that used the fallback)."""
    thr = threshold_source.groupby("cgid")["EventCount_t"].quantile(HIGH_LOAD_Q)
    pooled = float(threshold_source["EventCount_t"].quantile(HIGH_LOAD_Q))
    cg_thr = rows["cgid"].map(thr)
    n_fallback = int(rows.loc[cg_thr.isna(), "cgid"].nunique())
    cg_thr = cg_thr.fillna(pooled)

    order = rows.sort_values(["cgid", "stayid", "wstart"]).index
    r = rows.loc[order]
    high = (r["EventCount_t"].to_numpy() > cg_thr.loc[order].to_numpy())
    new_seq = np.r_[True, (r["cgid"].to_numpy()[1:] != r["cgid"].to_numpy()[:-1])
                    | (r["stayid"].to_numpy()[1:] != r["stayid"].to_numpy()[:-1])]
    streak = np.zeros(len(r))
    run = 0
    for i in range(len(r)):
        if new_seq[i]:
            run = 0
        run = run + 1 if high[i] else 0
        streak[i] = run
    return pd.Series(streak, index=order).loc[rows.index], n_fallback


def apply_train_only_streak(train, test, threshold_source=None, context=None):
    """Recompute the streak for train and test from a training-only
    threshold. Train and test rows of one stay are recomputed together so a
    run that crosses the split boundary is counted continuously, exactly as
    the upstream definition counts it; only the threshold changes.

    `context` (optional) supplies additional rows used only to keep a run
    continuous: the learning curve passes the full training set here while
    fitting the threshold on a subsample of it, so a test row is never
    separated from its true within-stay predecessors."""
    src = train if threshold_source is None else threshold_source
    ctx = train if context is None else context
    both = pd.concat([ctx, test])
    if not both.index.is_unique:
        raise ValueError("context must contain each row at most once")
    s, n_fb = streak_from_threshold(both, src)
    tr, te = train.copy(), test.copy()
    tr["ConsecutiveHighLoadCount_t"] = s.loc[tr.index].to_numpy()
    te["ConsecutiveHighLoadCount_t"] = s.loc[te.index].to_numpy()
    return tr, te, n_fb


# --------------------------------------------------------------- modelling
def fit_eval(train, test, feats):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(train[feats].to_numpy(float), train["y_next"].to_numpy(float))
    pred = m.predict(test[feats].to_numpy(float))
    y = test["y_next"].to_numpy(float)
    return (m, pred, float(mean_absolute_error(y, pred)),
            float(np.sqrt(mean_squared_error(y, pred))))


def clustered_ci(test, pred_ref, pred_new, n=N_BOOT, seed=SEED):
    """MAE(ref) - MAE(new), caregiver-clustered bootstrap. Positive => new
    better. Resamples caregivers with replacement; each caregiver contributes
    all of its test rows."""
    y = test["y_next"].to_numpy(float)
    d = np.abs(y - pred_ref) - np.abs(y - pred_new)
    codes, uniq = pd.factorize(test["cgid"].to_numpy())
    sums = np.bincount(codes, weights=d)
    cnts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(n, len(uniq)))
    boots = sums[picks].sum(axis=1) / cnts[picks].sum(axis=1)
    return float(d.mean()), (float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5)))


def row_ci(test, pred_ref, pred_new, n=N_BOOT, seed=SEED):
    y = test["y_next"].to_numpy(float)
    d = np.abs(y - pred_ref) - np.abs(y - pred_new)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                      for _ in range(n)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def shap_table(model, test, feats):
    import shap
    X = test[feats].to_numpy(float)
    sv = shap.TreeExplainer(model).shap_values(X)
    rows = []
    for j, f in enumerate(feats):
        fv = X[:, j]
        corr = (None if f in CATEGORICAL or np.std(fv) == 0
                else float(np.corrcoef(fv, sv[:, j])[0, 1]))
        rows.append({"feature": f, "mean_abs_shap": float(np.abs(sv[:, j]).mean()),
                     "value_shap_corr": corr})
    return sorted(rows, key=lambda r: -r["mean_abs_shap"])


def directions(table):
    out = {}
    for f, exp in PREREGISTERED.items():
        c = next((r["value_shap_corr"] for r in table if r["feature"] == f), None)
        if c is None:
            out[f] = {"expected": exp, "observed": "undefined", "corr": None,
                      "agrees": None}
        else:
            obs = "negative" if c < 0 else "positive"
            out[f] = {"expected": exp, "observed": obs, "corr": c,
                      "agrees": obs == exp}
    return out


def line(label, mae, rmse=None, d=None, ci=None, ref_gain=None):
    s = f"    {label:<34} MAE {mae:8.3f}"
    if rmse is not None:
        s += f"  RMSE {rmse:8.3f}"
    if d is not None:
        s += f"  dMAE vs A {d:+7.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
    if ref_gain:
        s += f"  {d / ref_gain:6.1%} of B's gain"
    print(s)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-learning-curve", action="store_true")
    ap.add_argument("--gate-tol", type=float, default=GATE_TOL,
                    help="tolerance for the verification gate. Leave at the "
                         "default. Widen only if the submitted figures cannot "
                         "be reproduced in any available environment; a run "
                         "with a widened gate is quotable only if EVERY ML "
                         "figure in the paper, Condition A included, is taken "
                         "from that same run.")
    args = ap.parse_args()
    t0 = time.time()
    summary, res = [], {"input": args.csv, "seed": SEED, "checks": {}}

    print("CCKG ML ablation v4\n" + "=" * 70)
    df = pd.read_csv(args.csv)
    panel, panel_all = build_panel(df)
    train0, test0 = temporal_split(panel)
    print(f"[setup] {len(df):,} rows from {args.csv}")
    print(f"[setup] panel {len(panel):,} rows, {panel.cgid.nunique():,} "
          f"caregivers >= {MIN_WINDOWS} windows; train {len(train0):,}  "
          f"test {len(test0):,}")
    res.update(n_panel=int(len(panel)), n_caregivers=int(panel.cgid.nunique()),
               n_train=int(len(train0)), n_test=int(len(test0)))

    # ---------------------------------------------------- [0] gate
    print("\n[0] VERIFICATION GATE: submitted specification "
          f"(must reproduce {PUBLISHED['A']:.3f} / {PUBLISHED['B']:.3f})")
    trL, teL = encode_legacy(panel_all, train0, test0)
    trL = trL.assign(ConsecutiveHighLoadCount_t=trL["streak_upstream"])
    teL = teL.assign(ConsecutiveHighLoadCount_t=teL["streak_upstream"])
    _, pA0, maeA0, rmseA0 = fit_eval(trL, teL, CORE)
    mB0, pB0, maeB0, rmseB0 = fit_eval(trL, teL, CORE + ONTO)
    d0, ci0 = clustered_ci(teL, pA0, pB0)
    line("A (submitted)", maeA0, rmseA0)
    line("B (submitted, full-set threshold)", maeB0, rmseB0, d0, ci0)
    ok = (abs(maeA0 - PUBLISHED["A"]) <= args.gate_tol
          and abs(maeB0 - PUBLISHED["B"]) <= args.gate_tol)
    res["gate_tolerance"] = args.gate_tol
    if args.gate_tol != GATE_TOL:
        print(f"    NOTE: gate tolerance widened to {args.gate_tol}. Quote figures "
              "from this run only if every ML figure in the paper, Condition A "
              "included, is replaced from this same run.")
    res["checks"]["gate_submitted_spec"] = {
        "MAE_A": maeA0, "RMSE_A": rmseA0, "MAE_B": maeB0, "RMSE_B": rmseB0,
        "delta": d0, "clustered_ci95": list(ci0), "reproduces_published": ok}
    summary.append({"check": "gate_submitted_spec", "MAE": maeB0, "RMSE": rmseB0,
                    "delta_vs_A": d0, "ci_low": ci0[0], "ci_high": ci0[1]})
    if not ok:
        sys.exit("    FAIL: the submitted figures do not reproduce from this file "
                 "and environment. Check the CSV and that xgboost is 3.3.0 (the "
                 "version the paper footnote names) before quoting any new "
                 "number. See the run instructions, step 4.")
    print("    PASS: submitted figures reproduce. New results are comparable.")

    # diagnostic: how much of the streak changes, and why
    full_s, _ = streak_from_threshold(panel, panel)
    agree_def = float((full_s.to_numpy() == panel["streak_upstream"].to_numpy()).mean())
    # streak with a training-only threshold, for the leakage-magnitude figure
    lk_s, _ = streak_from_threshold(pd.concat([train0, test0]), train0)
    lk_s = lk_s.loc[panel.index]
    changed_all = float((lk_s.to_numpy() != full_s.to_numpy()).mean())
    changed_test = float((lk_s.loc[test0.index].to_numpy()
                          != full_s.loc[test0.index].to_numpy()).mean())

    # ---------------------------------------------------- [1] new primary
    print("\n[1] NEW PRIMARY: streak threshold fitted on training windows only; "
          "encoder fitted on training rows")
    tr, te = encode(train0, test0)
    tr, te, n_fb = apply_train_only_streak(tr, te)
    both = pd.concat([tr, te]).loc[panel.index]
    agree_tr = float((both["ConsecutiveHighLoadCount_t"].to_numpy()
                      == panel["streak_upstream"].to_numpy()).mean())
    print(f"    leakage magnitude: the training-only threshold changes the streak "
          f"on {changed_all:.2%} of rows ({changed_test:.2%} of test rows) "
          f"relative to a threshold fitted on all panel windows")
    print(f"    agreement with the stored graph column: {agree_tr:.1%} "
          f"(full-panel threshold: {agree_def:.1%}; the remainder reflects the "
          f"threshold population, cohort-wide upstream versus modelling panel "
          f"here, not the train/test restriction)")
    print(f"    caregivers on pooled fallback threshold: {n_fb}")
    _, pA, maeA, rmseA = fit_eval(tr, te, CORE)
    mB, pB, maeB, rmseB = fit_eval(tr, te, CORE + ONTO)
    dB, ciB = clustered_ci(te, pA, pB)
    rciB = row_ci(te, pA, pB)
    line("A", maeA, rmseA)
    line("B", maeB, rmseB, dB, ciB)
    print(f"    reduction {dB / maeA:.1%};  row-level CI [{rciB[0]:+.2f}, "
          f"{rciB[1]:+.2f}] (for comparison only)")
    prim = {"MAE_A": maeA, "RMSE_A": rmseA, "MAE_B": maeB, "RMSE_B": rmseB,
            "delta": dB, "pct_gain": dB / maeA, "clustered_ci95": list(ciB),
            "row_ci95": list(rciB), "streak_agreement_with_upstream": agree_tr,
            "streak_agreement_definitional": agree_def,
            "leakage_rows_changed_all": changed_all,
            "leakage_rows_changed_test": changed_test,
            "caregivers_on_fallback_threshold": n_fb,
            "change_vs_submitted": {"MAE_B": maeB - maeB0, "delta": dB - d0}}
    if not args.skip_shap:
        tab = shap_table(mB, te, CORE + ONTO)
        dirs = directions(tab)
        onto_total = sum(r["mean_abs_shap"] for r in tab if r["feature"] in ONTO)
        print("    SHAP (mean |SHAP|, value-SHAP corr for continuous features):")
        for r in tab:
            c = "    --" if r["value_shap_corr"] is None else f"{r['value_shap_corr']:+.3f}"
            share = (f"  {r['mean_abs_shap'] / onto_total:5.1%} of ontology total"
                     if r["feature"] in ONTO else "")
            print(f"      {r['feature']:<28} {r['mean_abs_shap']:9.2f}  {c}{share}")
        for f, v in dirs.items():
            print(f"    {f}: pre-registered {v['expected']}, learned "
                  f"{v['observed']} ({v['corr']:+.3f}) -> "
                  f"{'AGREES' if v['agrees'] else 'DIFFERS'}")
        prim.update(shap=tab, directions=dirs)
        pd.DataFrame(tab).to_csv("ml_ablation_v4_shap.csv", index=False)
    res["checks"]["primary_train_only_streak"] = prim
    summary.append({"check": "A_new_primary", "MAE": maeA, "RMSE": rmseA})
    summary.append({"check": "B_new_primary", "MAE": maeB, "RMSE": rmseB,
                    "delta_vs_A": dB, "ci_low": ciB[0], "ci_high": ciB[1],
                    "pct_gain": dB / maeA})

    preds = te[["cgid", "stayid", "wstart"]].copy()
    preds["y_true_next_docInt"] = te["y_next"].to_numpy()
    preds["pred_conditionA"] = pA
    preds["pred_conditionB"] = pB
    preds["abs_err_A"] = np.abs(preds["y_true_next_docInt"] - pA)
    preds["abs_err_B"] = np.abs(preds["y_true_next_docInt"] - pB)
    assert abs(preds["abs_err_B"].mean() - maeB) < 1e-6
    preds.to_csv("ml_predictions_test_v4.csv", index=False)

    # ---------------------------------------------------- [2] baselines
    print("\n[2] BASELINES AND LEAVE-ONE-OUT (R2 #4), new primary specification")
    variants = [
        ("A + role", CORE + ["role_enc"]),
        ("A + role + unit", CORE + ["role_enc", "unit_enc"]),
        ("B - recovery gap", CORE + [f for f in ONTO if f != "InterWindowGap_t"]),
        ("B - streak", CORE + [f for f in ONTO if f != "ConsecutiveHighLoadCount_t"]),
        ("B - role", CORE + [f for f in ONTO if f != "role_enc"]),
        ("B - unit", CORE + [f for f in ONTO if f != "unit_enc"]),
    ]
    res["checks"]["baselines"] = {}
    for label, feats in variants:
        _, p, mae, rmse = fit_eval(tr, te, feats)
        d, ci = clustered_ci(te, pA, p)
        line(label, mae, rmse, d, ci, ref_gain=dB)
        res["checks"]["baselines"][label] = {
            "features": feats, "MAE": mae, "RMSE": rmse, "delta_vs_A": d,
            "clustered_ci95": list(ci), "share_of_B_gain": d / dB}
        summary.append({"check": label, "MAE": mae, "RMSE": rmse,
                        "delta_vs_A": d, "ci_low": ci[0], "ci_high": ci[1],
                        "share_of_B_gain": d / dB})

    # ---------------------------------------------------- [3] stay-grouped
    print("\n[3] STAY-GROUPED SPLIT (R3): no stay contributes to both train and test")
    trS, teS, single = stay_grouped_split(panel)
    trS, teS = encode(trS, teS)
    trS, teS, n_fbS = apply_train_only_streak(trS, teS)
    print(f"    train {len(trS):,}  test {len(teS):,}  caregivers "
          f"{teS.cgid.nunique():,}  (single-stay caregivers not splittable: "
          f"{single}; fallback thresholds: {n_fbS})")
    _, pAS, maeAS, rmseAS = fit_eval(trS, teS, CORE)
    mBS, pBS, maeBS, rmseBS = fit_eval(trS, teS, CORE + ONTO)
    dS, ciS = clustered_ci(teS, pAS, pBS)
    line("A (stay-grouped)", maeAS, rmseAS)
    line("B (stay-grouped)", maeBS, rmseBS, dS, ciS)
    print(f"    reduction {dS / maeAS:.1%}")
    stay = {"n_train": int(len(trS)), "n_test": int(len(teS)),
            "n_caregivers": int(teS.cgid.nunique()),
            "single_stay_caregivers_excluded": single,
            "MAE_A": maeAS, "RMSE_A": rmseAS, "MAE_B": maeBS, "RMSE_B": rmseBS,
            "delta": dS, "pct_gain": dS / maeAS, "clustered_ci95": list(ciS)}
    if not args.skip_shap:
        stay["directions"] = directions(shap_table(mBS, teS, CORE + ONTO))
        for f, v in stay["directions"].items():
            print(f"    {f}: learned {v['observed']} ({v['corr']:+.3f})")
    res["checks"]["stay_grouped_split"] = stay
    summary.append({"check": "B_stay_grouped", "MAE": maeBS, "RMSE": rmseBS,
                    "delta_vs_A": dS, "ci_low": ciS[0], "ci_high": ciS[1],
                    "pct_gain": dS / maeAS})

    # ---------------------------------------------------- [4] first window
    print("\n[4] FIRST-WINDOW INDICATOR, new primary")
    f4 = CORE + ONTO + ["first_window_t"]
    m4, p4, mae4, rmse4 = fit_eval(tr, te, f4)
    line("B + first-window indicator", mae4, rmse4)
    fw = {"MAE": mae4, "RMSE": rmse4,
          "first_window_share_of_test": float(te["first_window_t"].mean())}
    if not args.skip_shap:
        fw["directions"] = directions(shap_table(m4, te, f4))
        g = fw["directions"]["InterWindowGap_t"]
        print(f"    recovery-gap direction {g['corr']:+.3f} "
              f"({'unchanged' if g['agrees'] else 'SIGN CHANGED'}); first windows "
              f"are {fw['first_window_share_of_test']:.1%} of test")
    res["checks"]["first_window_indicator"] = fw
    summary.append({"check": "B_plus_first_window", "MAE": mae4, "RMSE": rmse4})

    # ---------------------------------------------------- [5] capacity
    print("\n[5] CAPACITY CONTROL: ontology columns permuted (matched marginals)")
    rng = np.random.default_rng(SEED)
    trC, teC, fake = tr.copy(), te.copy(), []
    for f in ONTO:
        trC[f"fake_{f}"] = rng.permutation(trC[f].to_numpy())
        teC[f"fake_{f}"] = rng.permutation(teC[f].to_numpy())
        fake.append(f"fake_{f}")
    _, pC, maeC, rmseC = fit_eval(trC, teC, CORE + fake)
    line("A + four permuted columns", maeC, rmseC)
    res["checks"]["capacity_shuffled"] = {"MAE": maeC, "RMSE": rmseC}
    summary.append({"check": "capacity_shuffled", "MAE": maeC, "RMSE": rmseC})

    # ---------------------------------------------------- [6] learning curve
    if not args.skip_learning_curve:
        print("\n[6] LEARNING CURVE, new primary (fixed test set; earliest frac of "
              "each caregiver's training rows; threshold refitted on the subsample)")
        lc = []
        ctx = encode(train0, test0)[0]      # sequence context only (see below)
        for frac in LC_FRACS:
            keep = []
            for _, g in train0.groupby("cgid", sort=False):
                keep.extend(g.index[:max(1, int(np.floor(len(g) * frac)))])
            sub, s_te = encode(train0.loc[keep], test0)
            # encoder and threshold fitted on the subsample only. The full
            # training rows serve as sequence context so a test row's run
            # length is counted from its true predecessors; no statistic is
            # taken from them.
            s_tr, s_te, nfb = apply_train_only_streak(sub, s_te, context=ctx)
            _, a, mA, _ = fit_eval(s_tr, s_te, CORE)
            _, b, mBv, _ = fit_eval(s_tr, s_te, CORE + ONTO)
            d, ci = clustered_ci(s_te, a, b, n=400)
            lc.append({"frac": frac, "n_train": int(len(sub)),
                       "n_caregivers": int(sub.cgid.nunique()), "MAE_A": mA,
                       "MAE_B": mBv, "delta": d, "ci_low": ci[0], "ci_high": ci[1],
                       "pct_gain": d / mA, "fallback_thresholds": nfb})
            print(f"    frac {frac:4.2f}  n_train {len(sub):>7,}  A {mA:7.2f}  "
                  f"B {mBv:7.2f}  dMAE {d:+6.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]  "
                  f"{d / mA:5.1%}")
        pd.DataFrame(lc).to_csv("ml_ablation_v4_learning_curve.csv", index=False)
        res["checks"]["learning_curve"] = lc

    res["environment"] = {"python": platform.python_version(),
                          "numpy": np.__version__, "pandas": pd.__version__,
                          "sklearn": sklearn.__version__,
                          "xgboost": xgb.__version__}
    json.dump(res, open("ml_ablation_v4_results.json", "w"), indent=2, default=float)
    pd.DataFrame(summary).to_csv("ml_ablation_v4_summary.csv", index=False)
    written = ["ml_ablation_v4_results.json", "ml_ablation_v4_summary.csv"]
    if not args.skip_shap:
        written.append("ml_ablation_v4_shap.csv")
    if not args.skip_learning_curve:
        written.append("ml_ablation_v4_learning_curve.csv")
    written.append("ml_predictions_test_v4.csv")
    print(f"\nwrote {', '.join(written)} ({time.time() - t0:.0f}s)")
    print("Quote SHAP figures from this run only; attributions vary across "
          "library versions, predictions do not.")


if __name__ == "__main__":
    main()
