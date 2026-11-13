"""CCKG Phase 2, Step 8: compute the well-being-risk score.

Reads the populated graph from Fuseki, computes a DETERMINISTIC,
theory-derived risk score per (caregiver, stay, window), and writes
WellbeingRiskAssessment individuals back out as Turtle.

    Risk = Demand x (1 - ResourceAdequacy) x (1 - RecoveryAdequacy)

grounded in Job Demands-Resources (demand-resource interaction) and
Effort-Recovery (recovery dimension). NOTHING HERE IS LEARNED. No
coefficient is fitted. See CCKG_RiskScore_and_Audit_Specification.md,
Part 1, for why a fitted form was rejected as circular.

=========================== READ BEFORE USING ===========================

1. THE RESOURCE TERM IS CONSTANT ON REAL DATA, AND THEREFORE CANCELS.
   MIMIC-III is single-institution and de-identified; no caregiver window
   can be attributed to a surveyed SOPS hospital. ResourceAdequacy is
   therefore one reference value applied to every row. Since
       Risk = (1 - k) * Demand * (1 - RecoveryAdequacy)
   for constant k, the resource term is a pure positive scalar: it scales
   every score identically and changes NO ranking, correlation, or
   ablation result on real data. It is retained because the schema and the
   theory require the three-term form, and because it DOES vary under the
   plasmode simulation. Report this explicitly; do not let a reader infer
   that SOPS resource variation is doing work on the MIMIC side.

2. POLARITY, NOT JDRClassification, DETERMINES ORIENTATION.
   JDRClassification says what a construct IS; Polarity says which way its
   number RUNS. They diverge: WorkplaceAggression is classified 'demand'
   but its measure is percent-positive on protection FROM aggression, so
   higher is better. Reading JDRClassification to orient a value inverts
   that construct. This script branches on Polarity only.

3. NORMALIZATION IS WITHIN-CAREGIVER, BY PERCENTILE RANK.
   Raw event counts are not comparable across caregivers or patients: a
   heavily monitored patient generates far more chart entries per window
   than a stable one. Each caregiver's windows are ranked against that same
   caregiver's own distribution. This reuses the convention already
   governing ConsecutiveHighLoadCount (per-caregiver 75th percentile),
   rather than introducing a second normalization scheme.

4. WINDOWS WITH NO PREDECESSOR ARE EXCLUDED, NOT IMPUTED.
   The first window of a (caregiver, stay) sequence has no InterWindowGap.
   Cross-stay gaps cannot be computed at all (per-patient date shifting).
   Those rows are excluded from scoring and COUNTED, and the count is
   reported as the coverage denominator. Imputing a gap would invent data.

5. CAREGIVERS WITH TOO FEW WINDOWS ARE EXCLUDED, NOT IMPUTED.
   A percentile rank over 1-3 observations is degenerate (n=1 gives 1.0 by
   construction). MIN_WINDOWS_PER_CAREGIVER is enforced and the excluded
   count is reported.

6. TWO NORMALIZATION VARIANTS ARE COMPUTED, AND THEY ARE NOT
   INTERCHANGEABLE.
   Variant A normalizes within caregiver. This controls the patient-acuity
   confound (a heavily monitored patient inflates every caregiver's counts)
   but forces each caregiver's demand distribution to uniform(0,1] BY
   CONSTRUCTION, so every caregiver ends with the same mean risk. Measured on
   the real cohort: between-caregiver variance share 0.0044, i.e. 99.6% of
   variation is within caregivers. Variant A answers "which of THIS
   caregiver's windows was worst" and CANNOT rank caregivers against each
   other.
   Variant B normalizes across the whole cohort. Between-caregiver variance
   share 0.2604, per-caregiver mean risk spread 0.0002-0.2144. Variant B can
   rank caregivers, but reintroduces the acuity confound that AcuityWeightedLoad
   (deferred, needs SOFA) is intended to remove.
   The two rank caregivers essentially independently: Spearman 0.0295 between
   their per-caregiver means. A downstream scheduling optimizer comparing
   clinicians needs B (or, better, an acuity-weighted cohort variant once SOFA
   is available); a within-clinician temporal signal needs A. Report which one
   any downstream result used.

Output: cckg-instances-risk-step8-v7.ttl
        step8_risk_scores_v7.csv       (same values, flat, for inspection)
=========================================================================
"""
import sys
from collections import Counter

import pandas as pd
import requests
from rdflib import Graph, Namespace, Literal, RDF, OWL, XSD

# ----------------------------------------------------------------- config
FUSEKI = "http://localhost:3030/cckg_v4/"   # note trailing slash; v4 dataset

# Which SOPS construct parameterises ResourceAdequacy. Staffing and Work Pace
# is the construct the two demand-side ProxyMappings target, so it is the
# defensible default. Changing this changes only the scalar k (see note 1).
RESOURCE_CONSTRUCT = "Staffing and Work Pace"

# Median, not mean: n=38 with observed outlier influence (Pearson -0.581 vs
# Spearman -0.753 on the staffing-burnout pair implies non-linearity/outliers).
RESOURCE_STATISTIC = "median"

# RECOVERY DEFINITION. Effort-Recovery theory says recovery FOLLOWS effort:
# exertion happens, then rest, and strain accumulates when the rest is too
# short. v1/v2 used the gap PRECEDING the window, which is the wrong side of
# the effort. Part 1 measured the consequence:
#   "preceding": corr(demand, recovery) = +0.247  -> the recovery term RISES
#                with demand and partially cancels the demand term it exists
#                to modulate; corr(risk, demand) only +0.627.
#   "following": corr(demand, recovery) = -0.313  -> behaves as theory
#                requires; corr(risk, demand) = +0.915.
# Coverage is unaffected: every window has either a preceding or a following
# gap except the first/last of each caregiver-stay sequence, and those counts
# are equal (322,945 each on the current corpus).
# "preceding" is retained ONLY so the paper can report the contrast.
RECOVERY_DEFINITION = "following"   # "following" (correct) | "preceding" (v1/v2)

# SYSTEM-ACCOUNT GUARD. MIMIC-III CGIDs are not all human clinicians. On the
# current corpus CGID 20889 appears in 23,256 of 23,259 distinct ICU stays
# (99.99%) across a 109-year shifted date span and supplies 30.8% of all
# scorable rows. No clinician touches every patient in a hospital's ICU
# history; this is an automated or default account. Including it in a
# CLINICIAN wellbeing analysis is indefensible regardless of numerical impact,
# and it also contaminates the cohort-normalised variant, whose percentiles
# would be built on a pool that is ~31% one non-human entity.
# Rule: a caregiver appearing in more than this share of all distinct stays is
# flagged, excluded, and REPORTED. Median real caregiver: 184 stays.
MAX_STAY_SHARE_PER_CAREGIVER = 0.50
EXCLUDE_SYSTEM_ACCOUNTS = True

MIN_WINDOWS_PER_CAREGIVER = 4

# Separate, stricter threshold for the CAREGIVER-LEVEL aggregate table. The
# per-window scores are unaffected by this. Ranking a caregiver with 10 windows
# against one with 1,670 on an unweighted mean is not defensible: the small-n
# caregiver's mean is dominated by sampling noise. Caregivers below this
# threshold are retained in the per-window output and EXCLUDED from the ranked
# aggregate, with the exclusion counted and reported.
MIN_WINDOWS_FOR_CAREGIVER_TABLE = 30

# Both normalization variants are computed. See note 6.
#   A "within_caregiver" : percentile within each caregiver's own windows.
#   B "cohort"           : percentile across all caregiver-windows.
VARIANTS = {"within_caregiver": "risk-v1-withinCG-multiplicative-JDR-ER",
            "cohort":           "risk-v1-cohort-multiplicative-JDR-ER"}

# Emitting both variants to RDF doubles the triple count (~4.5M each). Set to
# a single key to emit one, or keep both.
EMIT_VARIANTS = ["within_caregiver", "cohort"]

OUT_TTL = "cckg-instances-risk-step8-v7.ttl"
OUT_CSV = "step8_risk_scores_v7.csv"
OUT_CG_CSV = "step8_risk_by_caregiver_v7.csv"

CCKG = Namespace("http://knacc.umbc.edu/ontologies/cckg#")


# ------------------------------------------------------------------ fuseki
def sparql(query, label=""):
    """GET with the query as a URL parameter. The populated dataset is large;
    a feature-wide scan has been observed taking 100+ seconds, so the timeout
    is generous and progress is printed."""
    print(f"  querying Fuseki{(' [' + label + ']') if label else ''} ...",
          flush=True)
    resp = requests.get(
        FUSEKI,
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=600,
    )
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code} from {FUSEKI}", file=sys.stderr)
        print(resp.text[:500], file=sys.stderr)
        print("\nIf 404: check the dataset name in FUSEKI at the top of this "
              "script (it is cckg_v4, not cckg).", file=sys.stderr)
    resp.raise_for_status()
    return resp.json()["results"]["bindings"]


Q_FEATURES = """
PREFIX cckg: <http://knacc.umbc.edu/ontologies/cckg#>
SELECT ?cgid ?stayid ?wstart ?wtype ?docInt ?evCount ?gap ?role ?careUnit ?streak
WHERE {
  ?ea a cckg:EventAggregate ;
      cckg:aggregateOf     ?cg ;
      cckg:involvesStay    ?stay ;
      cckg:belongsToWindow ?w ;
      cckg:EventCount              ?evCount ;
      cckg:DocumentationIntensity  ?docInt ;
      cckg:ConsecutiveHighLoadCount ?streak .
  ?cg   cckg:CGID   ?cgid .
  ?stay cckg:StayID ?stayid .
  ?w    cckg:WindowStart ?wstart ; cckg:WindowType ?wtype .
  OPTIONAL { ?ea cckg:InterWindowGap ?gap }
  # role (CAREGIVERS.DESCRIPTION) and care-unit type are the two most
  # structurally informative categorical features. LEFT-joined so a missing
  # value never drops a row. ConsecutiveHighLoadCount is NOT optional: it is
  # always populated by the Step 2 population script (default 0, not absent),
  # unlike InterWindowGap which is genuinely absent for first-in-sequence
  # windows.
  OPTIONAL { ?cg   cckg:Role        ?role }
  OPTIONAL { ?stay cckg:occursInUnit/cckg:UnitName ?careUnit }
}
"""

Q_RESOURCE = """
PREFIX cckg: <http://knacc.umbc.edu/ontologies/cckg#>
SELECT ?construct ?polarity ?value
WHERE {
  ?h a cckg:HospitalContext ; cckg:hasSurveyEvent ?e .
  ?e cckg:hasConstructMeasurement ?m .
  ?m cckg:measuresConstruct ?c ;
     cckg:PercentPositive ?value ;
     cckg:Polarity ?polarity .
  ?c cckg:ConstructName ?construct .
}
"""


# ------------------------------------------------------------ pure scoring
def orient_to_adequacy(value, polarity):
    """Map a SOPS percent value onto [0,1] ADEQUACY, where 1 = fully adequate.

    Branches on Polarity, never on JDRClassification (see note 2). A
    higher_better construct maps straight through; a higher_worse construct
    (burnout, intent-to-leave: percent reporting a problem) is inverted.
    """
    # SCALE GUARD. A [0,1] range check on the OUTPUT cannot detect an
    # unscaled proportion: 0.5289/100 = 0.005289, still inside [0,1], silently
    # wrong by 100x. This is the defect that already occurred once in the SOPS
    # population (HSOPS 2.0 POS_* columns are proportions, not percents), so
    # the guard is on the INPUT and is explicit. The smallest value observed
    # across all 15 populated constructs is 13.24 (IntentToLeaveWPS), so any
    # value below 1.0 is a proportion that was never scaled.
    if not (0.0 <= value <= 100.0):
        raise ValueError(f"value {value} outside [0,100]; expected a percent")
    if 0.0 < value < 1.0:
        raise ValueError(
            f"value {value} looks like a PROPORTION, not a percent. No "
            f"populated SOPS construct falls below 13.24. Check ValueScale on "
            f"the ConstructMeasurement and re-run the population script.")

    if polarity == "higher_better":
        adequacy = value / 100.0
    elif polarity == "higher_worse":
        adequacy = 1.0 - (value / 100.0)
    else:
        raise ValueError(f"unknown Polarity {polarity!r}; expected "
                         f"'higher_better' or 'higher_worse'")
    return adequacy


def within_caregiver_percentile(df, column, group="cgid"):
    """Percentile rank of `column` within each caregiver's own distribution.

    Ties averaged. Returns values in (0,1]. Callers must have already
    filtered groups smaller than MIN_WINDOWS_PER_CAREGIVER, since a rank
    over n=1 is 1.0 by construction and carries no information.
    """
    return df.groupby(group)[column].rank(method="average", pct=True)


def compute_risk(demand, resource_adequacy, recovery_adequacy):
    """Risk = Demand x (1 - ResourceAdequacy) x (1 - RecoveryAdequacy).

    Multiplicative, not additive: JD-R holds that adequate resources and
    adequate recovery each INDEPENDENTLY gate whether demand becomes strain,
    so adequacy on either dimension must drive the score toward zero
    regardless of demand. An additive form would let resources offset demand,
    which the theory does not claim.
    """
    return demand * (1.0 - resource_adequacy) * (1.0 - recovery_adequacy)


def between_caregiver_variance_share(df, column, group="cgid"):
    """SS_between / SS_total for `column`, grouped by caregiver.

    1.0 = all variation is between caregivers; 0.0 = none is, i.e. the score
    cannot discriminate caregivers at all. Reported for both variants because
    it is the single number that decides whether a score is usable as a
    cross-clinician optimizer input.
    """
    grand = df[column].mean()
    ss_between = (df.groupby(group)[column].transform("mean") - grand).pow(2).sum()
    ss_total = (df[column] - grand).pow(2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


# ------------------------------------------------------------------- main
def main():
    print("Step 8: well-being-risk score\n" + "=" * 60)

    # --- resource term ---------------------------------------------------
    print("\n[1/5] resource reference value")
    rrows = sparql(Q_RESOURCE, "SOPS constructs")
    if not rrows:
        raise SystemExit(
            "No ConstructMeasurement data returned. Confirm "
            "cckg-instances-sops-constructs-v4.ttl is loaded into the dataset "
            "named in FUSEKI.")
    res = pd.DataFrame([{
        "construct": r["construct"]["value"],
        "polarity": r["polarity"]["value"],
        "value": float(r["value"]["value"]),
    } for r in rrows])

    sel = res[res.construct == RESOURCE_CONSTRUCT]
    if sel.empty:
        raise SystemExit(
            f"Construct {RESOURCE_CONSTRUCT!r} not found. Available:\n  "
            + "\n  ".join(sorted(res.construct.unique())))
    polarity = sel.polarity.iloc[0]
    raw = sel.value.median() if RESOURCE_STATISTIC == "median" else sel.value.mean()
    resource_adequacy = orient_to_adequacy(raw, polarity)
    print(f"  {RESOURCE_CONSTRUCT}: n={len(sel)} hospitals, "
          f"{RESOURCE_STATISTIC}={raw:.4f}, polarity={polarity}")
    print(f"  ResourceAdequacy = {resource_adequacy:.4f}  (CONSTANT for every "
          f"row; scales all scores, changes no ranking -- see note 1)")

    # --- features --------------------------------------------------------
    print("\n[2/5] caregiver-window features")
    frows = sparql(Q_FEATURES, "EventAggregate")
    if not frows:
        raise SystemExit(
            "No EventAggregate data returned. Confirm the MIMIC step1/step2 "
            "TTL files are loaded into this dataset.")
    df = pd.DataFrame([{
        "cgid": int(r["cgid"]["value"]),
        "stayid": int(r["stayid"]["value"]),
        "wstart": r["wstart"]["value"],
        "wtype": r["wtype"]["value"],
        "docInt": float(r["docInt"]["value"]),
        "evCount": int(r["evCount"]["value"]),
        "gap": float(r["gap"]["value"]) if "gap" in r else None,
        "role": r["role"]["value"] if "role" in r else None,
        "careUnit": r["careUnit"]["value"] if "careUnit" in r else None,
        "consecutiveHighLoad": int(r["streak"]["value"]),
    } for r in frows])
    n_total = len(df)
    role_cov = 100.0 * df["role"].notna().mean()
    unit_cov = 100.0 * df["careUnit"].notna().mean()
    print(f"  role populated:      {role_cov:.1f}% of rows "
          f"({df['role'].nunique()} distinct)")
    print(f"  care unit populated: {unit_cov:.1f}% of rows "
          f"({df['careUnit'].nunique()} distinct)")
    print(f"  {n_total:,} caregiver-window rows, "
          f"{df.cgid.nunique():,} distinct caregivers")

    # --- exclusions ------------------------------------------------------
    print("\n[3/5] exclusions (reported, not imputed)")
    excl = Counter()

    # Derive the recovery interval FIRST, then exclude on the one actually
    # used. Excluding on the preceding gap and then using the following gap
    # would compound two alternative criteria and halve coverage for no reason.
    df = df.sort_values(["cgid", "stayid", "wstart"]).copy()
    if RECOVERY_DEFINITION == "following":
        df["recovery_gap"] = df.groupby(["cgid", "stayid"]).gap.shift(-1)
        why = "no following gap (last window of a caregiver-stay sequence)"
    elif RECOVERY_DEFINITION == "preceding":
        df["recovery_gap"] = df["gap"]
        why = "no preceding gap (first window of a caregiver-stay sequence)"
    else:
        raise ValueError(f"unknown RECOVERY_DEFINITION {RECOVERY_DEFINITION!r}")
    print(f"  recovery interval = gap {RECOVERY_DEFINITION} the window")

    no_gap = df.recovery_gap.isna().sum()
    excl[why] = int(no_gap)
    df = df[df.recovery_gap.notna()].copy()

    if EXCLUDE_SYSTEM_ACCOUNTS:
        n_stays_total = df.stayid.nunique()
        per_cg_stays = df.groupby("cgid").stayid.nunique()
        sysaccts = per_cg_stays[
            per_cg_stays > MAX_STAY_SHARE_PER_CAREGIVER * n_stays_total].index
        n_sys_rows = int(df.cgid.isin(sysaccts).sum())
        excl[f"system account (appears in >"
             f"{100*MAX_STAY_SHARE_PER_CAREGIVER:.0f}% of all stays)"] = n_sys_rows
        for cg in sysaccts:
            print(f"    FLAGGED CGID {cg}: {per_cg_stays[cg]:,} of "
                  f"{n_stays_total:,} stays "
                  f"({100*per_cg_stays[cg]/n_stays_total:.2f}%) -- excluded. "
                  f"Verify its CAREGIVERS.DESCRIPTION and report the exclusion.")
        df = df[~df.cgid.isin(sysaccts)].copy()

    counts = df.groupby("cgid").size()
    too_few = counts[counts < MIN_WINDOWS_PER_CAREGIVER].index
    n_too_few = int(df.cgid.isin(too_few).sum())
    excl[f"caregiver has < {MIN_WINDOWS_PER_CAREGIVER} scorable windows"] = n_too_few
    df = df[~df.cgid.isin(too_few)].copy()

    for reason, n in excl.items():
        print(f"  excluded {n:>8,}  {reason}")
    n_cg_total = pd.DataFrame([{"cgid": int(r["cgid"]["value"])}
                               for r in frows]).cgid.nunique()
    print(f"  scorable {len(df):>8,} of {n_total:,} rows "
          f"({100.0 * len(df) / n_total:.1f}% ROW coverage)")
    print(f"  scorable {df.cgid.nunique():>8,} of {n_cg_total:,} caregivers "
          f"({100.0 * df.cgid.nunique() / n_cg_total:.1f}% CAREGIVER coverage)")
    print("  NOTE: caregiver coverage is the lower and more conservative "
          "figure; report both. A caregiver whose every window is a "
          "first-window is dropped entirely.")
    if df.empty:
        raise SystemExit("Nothing scorable after exclusions.")

    # --- score -----------------------------------------------------------
    print("\n[4/5] scoring (both normalization variants)")
    df["resource_adequacy"] = resource_adequacy
    # Longer gap = more recovery = higher adequacy, so the percentile of the
    # gap IS the adequacy; no inversion.
    for variant in VARIANTS:
        if variant == "within_caregiver":
            dem = within_caregiver_percentile(df, "docInt")
            rec = within_caregiver_percentile(df, "recovery_gap")
        elif variant == "cohort":
            dem = df["docInt"].rank(method="average", pct=True)
            rec = df["recovery_gap"].rank(method="average", pct=True)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        df[f"demand_{variant}"] = dem
        df[f"recovery_{variant}"] = rec
        df[f"risk_{variant}"] = compute_risk(dem, resource_adequacy, rec)

        r = df[f"risk_{variant}"]
        assert r.between(0, 1).all(), f"risk outside [0,1] for {variant}"
        assert dem.between(0, 1).all(), f"demand outside [0,1] for {variant}"
        assert rec.between(0, 1).all(), f"recovery outside [0,1] for {variant}"

        share = between_caregiver_variance_share(df, f"risk_{variant}")
        cg_means = df.groupby("cgid")[f"risk_{variant}"].mean()
        print(f"  [{variant}]")
        print(f"    risk      min={r.min():.4f} median={r.median():.4f} "
              f"max={r.max():.4f} mean={r.mean():.4f}")
        print(f"    per-caregiver mean risk: sd={cg_means.std():.4f} "
              f"range={cg_means.min():.4f}-{cg_means.max():.4f}")
        print(f"    BETWEEN-CAREGIVER VARIANCE SHARE = {share:.4f}"
              + ("   <-- cannot rank caregivers" if share < 0.05 else
                 "   <-- can rank caregivers"))

    # Effective ceiling: the constant resource term caps the score well below
    # 1.0, so a raw value must not be read as a proportion of maximum risk.
    print(f"  effective max given constant resource: "
          f"{1.0 * (1.0 - resource_adequacy):.4f} (NOT 1.0)")

    # DIAGNOSTIC that would have caught the v1/v2 error: the recovery term
    # must not rise with demand. A positive value here means the two terms of
    # the formula are fighting each other.
    dr = df.groupby("cgid").apply(
        lambda g: g["docInt"].corr(g["recovery_within_caregiver"],
                                   method="spearman"), include_groups=False)
    dr = dr.dropna()
    print(f"  corr(demand, recovery) median across caregivers: {dr.median():+.4f}"
          + ("   <-- WRONG SIGN, recovery rises with demand"
             if dr.median() > 0.05 else "   OK"))

    # Do the two variants order caregivers the same way?
    ma = df.groupby("cgid")["risk_within_caregiver"].mean()
    mb = df.groupby("cgid")["risk_cohort"].mean()
    print(f"  rank corr of per-caregiver means, A vs B: "
          f"{ma.corr(mb, method='spearman'):.4f}")

    # Sanity: demand must be a monotone transform of docInt within caregiver.
    # NOTE: a caregiver with a single distinct docInt gives a nan correlation;
    # nan < 0.999 is False, so v1 of this check counted those as passing. They
    # are now counted separately rather than silently absorbed.
    bad, undefined = 0, 0
    for _, grp in df.groupby("cgid"):
        c = grp.docInt.corr(grp["demand_within_caregiver"], method="spearman")
        if pd.isna(c):
            undefined += 1
        elif c < 0.999:
            bad += 1
    n_cg = df.cgid.nunique()
    print(f"  demand-vs-docInt rank agreement: {n_cg - bad - undefined}/{n_cg} "
          f"exact, {undefined} undefined (constant docInt), {bad} mismatched"
          + ("  <-- INVESTIGATE" if bad else ""))

    # --- emit ------------------------------------------------------------
    print("\n[5/5] writing")
    g = Graph()
    g.bind("cckg", CCKG)
    g.bind("owl", OWL)
    for variant in EMIT_VARIANTS:
        fv = VARIANTS[variant]
        for row in df.itertuples(index=False):
            # URI encodes caregiver + stay + window + variant. assessedIn points
            # at the ActivityWindow (per-stay, not per-caregiver) and
            # assessmentOf at the Caregiver; together they identify the source
            # EventAggregate. NOTE: the schema has no direct
            # WellbeingRiskAssessment -> EventAggregate edge, so provenance is
            # recoverable by join but not asserted. Flagged for schema v5.
            # The two variants are distinguished by FormulaVersion, which the
            # schema already carries, so no schema change is needed to hold both.
            # v7 FIX. The ActivityWindow URI must match EXACTLY the identifier
            # minted by populate_mimic_step2.py, which is:
            #     wid = f"AW_{stay}_{wstart.strftime('%Y%m%dT%H%M')}"
            # i.e. prefix "AW_", a literal "T" separator, and NO seconds.
            # v6 and earlier built "ActivityWindow_<stay>_<YYYYMMDD>_<HHMMSS>"
            # from an assumed convention that was never checked against the
            # population script. Every assessedIn triple therefore pointed at a
            # non-existent node: the risk file loaded without error and joined
            # to nothing. Derive the tag from the same source instead of
            # reconstructing it by string surgery.
            _ts = pd.Timestamp(row.wstart)
            wtag = _ts.strftime("%Y%m%dT%H%M")          # matches step2 exactly
            risk_tag = _ts.strftime("%Y%m%d_%H%M")      # for the risk node's own URI
            tag = "W" if variant == "within_caregiver" else "C"
            u = CCKG[f"Risk{tag}_{row.cgid}_{row.stayid}_{risk_tag}"]
            g.add((u, RDF.type, CCKG.WellbeingRiskAssessment))
            g.add((u, RDF.type, OWL.NamedIndividual))
            g.add((u, CCKG.assessmentOf, CCKG[f"Caregiver_{row.cgid}"]))
            g.add((u, CCKG.assessedIn,
                   CCKG[f"AW_{row.stayid}_{wtag}"]))
            g.add((u, CCKG.RiskScore, Literal(
                round(getattr(row, f"risk_{variant}"), 6), datatype=XSD.decimal)))
            g.add((u, CCKG.DemandInput, Literal(
                round(getattr(row, f"demand_{variant}"), 6), datatype=XSD.decimal)))
            g.add((u, CCKG.ResourceInput, Literal(
                round(row.resource_adequacy, 6), datatype=XSD.decimal)))
            g.add((u, CCKG.RecoveryInput, Literal(
                round(getattr(row, f"recovery_{variant}"), 6), datatype=XSD.decimal)))
            g.add((u, CCKG.FormulaVersion, Literal(fv)))

    # Caregiver-level aggregate: this, not the per-window score, is the shape a
    # scheduling optimizer consumes as a burden or fairness term. Emitted as CSV
    # only (no schema class exists for a caregiver-level summary).
    agg_all = df.groupby("cgid").agg(
        n_windows=("risk_cohort", "size"),
        mean_risk_within=("risk_within_caregiver", "mean"),
        mean_risk_cohort=("risk_cohort", "mean"),
        p90_risk_cohort=("risk_cohort", lambda s: s.quantile(0.90)),
        mean_docInt=("docInt", "mean"),
        median_recovery_gap=("recovery_gap", "median"),
    ).reset_index()
    n_before = len(agg_all)
    agg = agg_all[agg_all.n_windows >= MIN_WINDOWS_FOR_CAREGIVER_TABLE].copy()
    agg = agg.sort_values("mean_risk_cohort", ascending=False)
    agg.to_csv(OUT_CG_CSV, index=False)
    print(f"  caregiver table: {len(agg):,} of {n_before:,} caregivers retained "
          f"(>= {MIN_WINDOWS_FOR_CAREGIVER_TABLE} windows); "
          f"{n_before - len(agg):,} excluded as too few windows to rank")

    # v7: verify every assessedIn target is a URI this pipeline could have minted,
    # and print one example so it can be eyeballed against the step2 file. A
    # dangling object reference does not raise on load -- it just joins to
    # nothing -- so it must be checked here.
    _targets = {str(o) for _, _, o in g.triples((None, CCKG.assessedIn, None))}
    _example = sorted(_targets)[0] if _targets else "(none)"
    print(f"  assessedIn targets: {len(_targets):,} distinct")
    print(f"  example ActivityWindow URI emitted: {_example}")
    print("  ^ this MUST match an ActivityWindow individual in "
          "cckg-instances-mimic-step2.ttl (pattern AW_<stay>_<YYYYMMDDTHHMM>)")

    g.serialize(OUT_TTL, format="turtle")
    df.to_csv(OUT_CSV, index=False)
    print(f"  {len(g):,} triples -> {OUT_TTL} "
          f"({len(EMIT_VARIANTS)} variant(s))")
    print(f"  {len(df):,} rows    -> {OUT_CSV}")
    print(f"  {len(agg):,} caregivers -> {OUT_CG_CSV}")
    print("\nCOVERAGE (Phase 3 evaluation metric): "
          f"{100.0 * len(df) / n_total:.1f}% of caregiver-window combinations "
          "have a computable risk score.")
    print("\nWARNING TO CARRY INTO SECTION IV: ResourceAdequacy is constant "
          f"({resource_adequacy:.4f}) on real MIMIC data. It scales every "
          "score by the same factor and changes no ranking. The three-term "
          "form is architecturally correct and theory-faithful, but on this "
          "corpus it is numerically a two-term demand-recovery score.")


if __name__ == "__main__":
    main()
