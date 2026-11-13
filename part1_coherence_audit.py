
import sys

import numpy as np
import pandas as pd
import requests
from scipy import stats

FUSEKI = "http://localhost:3030/cckg_v4/"
STEP8_CSV = "step8_risk_scores.csv"
SYSTEM_ACCOUNT = 20889
OUT_CSV = "part1_coherence_results.csv"
MIN_N = 10          # windows per caregiver for that caregiver to contribute
OUT_FIG_NOTE = True

Q_EXTRA = """
PREFIX cckg: <http://knacc.umbc.edu/ontologies/cckg#>
SELECT ?cgid ?stayid ?wstart ?streak ?procCount
WHERE {
  ?ea a cckg:EventAggregate ;
      cckg:aggregateOf     ?cg ;
      cckg:involvesStay    ?stay ;
      cckg:belongsToWindow ?w ;
      cckg:ConsecutiveHighLoadCount ?streak ;
      cckg:ProcedureCount           ?procCount .
  ?cg   cckg:CGID   ?cgid .
  ?stay cckg:StayID ?stayid .
  ?w    cckg:WindowStart ?wstart .
}
"""


def sparql(query, label=""):
    print(f"  querying Fuseki [{label}] ...", flush=True)
    r = requests.get(FUSEKI, params={"query": query},
                     headers={"Accept": "application/sparql-results+json"},
                     timeout=900)
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def within_caregiver_rho(df, a, b, min_n=MIN_N):
    """Spearman rho of a vs b computed separately inside each caregiver.

    Returns (array of rho, n_excluded_small, n_excluded_novar). A caregiver
    with no variance on either variable yields an undefined coefficient and is
    counted, not silently absorbed as a pass (that defect occurred once in the
    Step 8 sanity check).
    """
    rhos, small, novar = [], 0, 0
    for _, g in df.groupby("cgid"):
        if len(g) < min_n:
            small += 1
            continue
        if g[a].nunique() < 2 or g[b].nunique() < 2:
            novar += 1
            continue
        rho, _ = stats.spearmanr(g[a], g[b])
        if np.isnan(rho):
            novar += 1
        else:
            rhos.append(rho)
    return np.array(rhos), small, novar


def report(hyp_id, name, arr, predicted, small, novar, notes=""):
    med = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    frac = float((arr < 0).mean() if predicted == "negative" else (arr > 0).mean())
    _, p = stats.wilcoxon(arr)
    observed = "negative" if med < 0 else "positive"
    verdict = "HOLDS" if observed == predicted else "FAILS"
    print(f"  [{hyp_id}] {name}")
    print(f"      predicted {predicted:8s} observed {observed:8s} -> {verdict}")
    print(f"      median rho {med:+.4f}  IQR [{q1:+.3f}, {q3:+.3f}]  "
          f"n_caregivers={len(arr)}")
    print(f"      caregivers in predicted direction: {100 * frac:.1f}%")
    print(f"      Wilcoxon vs 0: p={p:.3g}   excluded: {small} small, "
          f"{novar} no-variance")
    if notes:
        print(f"      NOTE: {notes}")
    print()
    return dict(hypothesis=hyp_id, test=name, predicted=predicted,
                observed=observed, verdict=verdict, median_rho=med,
                iqr_low=q1, iqr_high=q3, n_caregivers=len(arr),
                pct_in_predicted_direction=100 * frac, wilcoxon_p=p,
                excluded_small=small, excluded_novariance=novar, notes=notes)


def main():
    print("Part 1: within-MIMIC operational coherence\n" + "=" * 70)

    import argparse
    global FUSEKI, STEP8_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=STEP8_CSV)
    ap.add_argument("--endpoint", default=FUSEKI)
    args = ap.parse_args()
    FUSEKI, STEP8_CSV = args.endpoint, args.csv

    df = pd.read_csv(STEP8_CSV)
    if "recovery_gap" in df.columns:
        sys.exit("FATAL: this is a v6/v7 Step 8 export. Part 1 needs the "
                 "first-generation export (step8_risk_scores.csv from "
                 "compute_wellbeing_risk_step8.py); see the docstring.")
    n_sys = int((df.cgid == SYSTEM_ACCOUNT).sum())
    df = df[df.cgid != SYSTEM_ACCOUNT].copy()
    print(f"\n[1/3] {len(df):,} rows, {df.cgid.nunique():,} caregivers "
          f"from {STEP8_CSV} (system account CGID {SYSTEM_ACCOUNT}: "
          f"{n_sys:,} rows excluded)")

    print("\n[2/3] pulling ConsecutiveHighLoadCount / ProcedureCount")
    try:
        rows = sparql(Q_EXTRA, "streak+procedures")
        extra = pd.DataFrame([{
            "cgid": int(r["cgid"]["value"]),
            "stayid": int(r["stayid"]["value"]),
            "wstart": r["wstart"]["value"],
            "streak": int(r["streak"]["value"]),
            "procCount": int(r["procCount"]["value"]),
        } for r in rows])
        df = df.merge(extra, on=["cgid", "stayid", "wstart"], how="left")
        have_streak = df.streak.notna().all()
        print(f"  merged {len(extra):,} rows; streak available: {have_streak}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  WARNING: could not pull extra features ({e}).")
        print("  H4/H5 will be skipped; H1-H3 do not need them.")
        have_streak = False

    print("\n[3/3] hypotheses\n")
    results = []

    # H1 --------------------------------------------------------------
    a, s, nv = within_caregiver_rho(df, "docInt", "gap")
    results.append(report(
        "H1", "DocumentationIntensity vs InterWindowGap (same window)",
        a, "negative", s, nv,
        "InterWindowGap is the interval BEFORE the window. A negative "
        "coefficient was predicted on the reading that a long gap means rest. "
        "See the restricted re-test below before interpreting."))

    # H1b: restricted to genuine gaps -----------------------------------
    nz = df[df.gap > 0]
    a2, s2, nv2 = within_caregiver_rho(nz, "docInt", "gap")
    results.append(report(
        "H1b", "same, restricted to windows with a non-zero gap",
        a2, "negative", s2, nv2,
        f"{100 * len(nz) / len(df):.1f}% of rows have gap>0. The pooled H1 "
        "result is dominated by the ~78% of rows where gap is exactly 0 "
        "(back-to-back windows)."))

    # H2 --------------------------------------------------------------
    a3, s3, nv3 = within_caregiver_rho(df, "docInt", "evCount")
    results.append(report(
        "H2", "DocumentationIntensity vs EventCount (convergent validity)",
        a3, "positive", s3, nv3,
        "Near-perfect agreement is expected and is a redundancy check, not an "
        "independent finding: EventCount includes documentation events."))

    # H3 --------------------------------------------------------------
    d2 = df.sort_values(["cgid", "stayid", "wstart"]).copy()
    d2["next_docInt"] = d2.groupby(["cgid", "stayid"]).docInt.shift(-1)
    d3 = d2.dropna(subset=["next_docInt"])
    a4, s4, nv4 = within_caregiver_rho(d3, "gap", "next_docInt")
    results.append(report(
        "H3", "InterWindowGap vs NEXT-window demand (lagged Effort-Recovery)",
        a4, "negative", s4, nv4,
        "Tests whether insufficient recovery PRECEDES elevated demand, which "
        "is the directional form Effort-Recovery actually claims."))

    # H4 / H5 ----------------------------------------------------------
    if have_streak:
        a5, s5, nv5 = within_caregiver_rho(df, "streak", "gap")
        results.append(report(
            "H4", "ConsecutiveHighLoadCount vs InterWindowGap",
            a5, "negative", s5, nv5,
            "A high-load streak should coincide with short recovery intervals."))

        a6, s6, nv6 = within_caregiver_rho(df, "streak", "docInt")
        results.append(report(
            "H5", "ConsecutiveHighLoadCount vs DocumentationIntensity",
            a6, "positive", s6, nv6,
            "PARTIALLY CIRCULAR: the streak flag is derived from a percentile "
            "of documentation intensity, so a positive coefficient is largely "
            "definitional. Reported for completeness, not as evidence."))

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(out)} tests)")
    print("\nReport in Section IV as: median rho across caregivers, IQR, and "
          "percentage of caregivers in the predicted direction. Do NOT report "
          "pooled coefficients or their p-values.")


if __name__ == "__main__":
    main()
