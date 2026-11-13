
import argparse
import sys

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ENDPOINT = "http://localhost:3030/cckg_v4/"
#WPS_CSV = 
#H20_CSV = 

# (label, construct X, construct Y, predicted sign, published rho, published n)
PAIRS = [
    ("Staffing ~ Burnout",        "StaffingAndWorkPace",            "BurnoutWPG1",      -1, -0.75, 38),
    ("Staffing ~ Intent-to-leave","StaffingAndWorkPace",            "IntentToLeaveWPS", -1, -0.72, 36),
    ("Handoffs ~ Burnout",        "HandoffsInformationExchange",    "BurnoutWPG1",      -1, -0.45, 38),
    ("Mgmt support ~ Burnout",    "ManagementSupportPatientSafety", "BurnoutWPG1",      -1, -0.51, 38),
    ("Burnout ~ Intent-to-leave", "BurnoutWPG1",                    "IntentToLeaveWPS", +1, +0.65, 36),
]

# construct individual -> (file key, source column); mirrors
# build_sops_construct_measurements_v4.py
CSV_COLUMN = {
    "StaffingAndWorkPace":            ("H20", "POS_STAFF"),
    "HandoffsInformationExchange":    ("H20", "POS_HANDOFF"),
    "ManagementSupportPatientSafety": ("H20", "POS_MGMT"),
    "BurnoutWPG1":                    ("WPS", "burnout_pct_3plus"),
    "IntentToLeaveWPS":               ("WPS", "leave_pct_any_yes"),
}

Q_PAIR = """
PREFIX cckg: <http://knacc.umbc.edu/ontologies/cckg#>
SELECT ?publicID ?x ?y WHERE {{
  ?h a cckg:HospitalContext ; cckg:PublicID ?publicID ;
     cckg:hasSurveyEvent ?ex , ?ey .
  ?ex cckg:hasConstructMeasurement ?mx .
  ?mx cckg:measuresConstruct cckg:{cx} ; cckg:PercentPositive ?x .
  ?ey cckg:hasConstructMeasurement ?my .
  ?my cckg:measuresConstruct cckg:{cy} ; cckg:PercentPositive ?y .
}}
"""


def pairs_from_graph(endpoint, cx, cy):
    import requests
    r = requests.get(endpoint, params={"query": Q_PAIR.format(cx=cx, cy=cy)},
                     headers={"Accept": "application/sparql-results+json"},
                     timeout=300)
    r.raise_for_status()
    rows = [(int(b["publicID"]["value"]), float(b["x"]["value"]),
             float(b["y"]["value"])) for b in r.json()["results"]["bindings"]]
    df = pd.DataFrame(rows, columns=["PUBLICID", "x", "y"])
    dup = df.duplicated("PUBLICID").sum()
    if dup:
        # A duplicated ConstructName or a second constructs file loaded into
        # the same dataset doubles rows silently (data bug #1). Fail loudly.
        sys.exit(f"FATAL: {dup} duplicate hospital rows for {cx}~{cy}. "
                 "Check that exactly one constructs TTL (v4) is loaded.")
    return df


def pairs_from_csv(cx, cy):
    wps = pd.read_csv(WPS_CSV)
    h20 = pd.read_csv(H20_CSV)
    for d in (wps, h20):
        d["PUBLICID"] = d["PUBLICID"].astype(int)
    ids = sorted(set(wps.PUBLICID) & set(h20.PUBLICID))
    assert len(ids) == 38, f"expected 38 joined hospitals, got {len(ids)}"
    src = {"WPS": wps.set_index("PUBLICID").loc[ids],
           "H20": h20.set_index("PUBLICID").loc[ids]}
    fx, colx = CSV_COLUMN[cx]
    fy, coly = CSV_COLUMN[cy]
    df = pd.DataFrame({"x": src[fx][colx], "y": src[fy][coly]})
    return df.dropna().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["graph", "csv"], default="graph")
    ap.add_argument("--endpoint", default=ENDPOINT)
    a = ap.parse_args()

    print(f"Part 2: within-SOPS construct coherence (source: {a.source})\n"
          + "=" * 70)
    rows, all_match = [], True
    for label, cx, cy, sign, pub_rho, pub_n in PAIRS:
        df = (pairs_from_graph(a.endpoint, cx, cy) if a.source == "graph"
              else pairs_from_csv(cx, cy))
        rho = spearmanr(df.x, df.y).statistic
        r = pearsonr(df.x, df.y)[0]
        holds = (rho < 0) == (sign < 0)
        match = round(rho, 2) == pub_rho and len(df) == pub_n
        all_match &= match
        print(f"  {label:<27} n={len(df):<3} Spearman {rho:+.3f}  "
              f"(Pearson {r:+.3f})  {'holds' if holds else 'FAILS'}   "
              f"paper {pub_rho:+.2f}, n={pub_n}: "
              f"{'match' if match else 'MISMATCH'}")
        rows.append({"pair": label, "construct_x": cx, "construct_y": cy,
                     "predicted": "negative" if sign < 0 else "positive",
                     "n": len(df), "spearman": rho, "pearson": r,
                     "verdict": "holds" if holds else "fails",
                     "matches_paper": match})
    pd.DataFrame(rows).to_csv("part2_coherence_results.csv", index=False)
    print("\nwrote part2_coherence_results.csv")
    if not all_match:
        sys.exit("At least one coefficient does not reproduce Table II.")
    print("All five coefficients reproduce Table II.")


if __name__ == "__main__":
    main()
