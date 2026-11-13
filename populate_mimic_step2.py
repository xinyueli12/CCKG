"""Populate the CCKG operational layer, Step 2: ActivityWindow and
EventAggregate from CHARTEVENTS, PROCEDUREEVENTS_MV, DATETIMEEVENTS.

Run from the folder containing this script. Adjust MIMIC3_DIR below.
Output: cckg-instances-mimic-step2.ttl

Reuses the SAME Caregiver_/ICUStay_ URIs as populate_mimic_step1.py, so
when both files are loaded into the same Fuseki dataset, the object
properties (aggregateOf, involvesStay) resolve against the individuals
Step 1 already created.

======================= SCOPE DECISIONS, READ FIRST =======================

1. UniqueStayCount is NOT populated. The schema comment describes it as
   "distinct ICUSTAY_ID per CGID-window", which implicitly asks whether a
   caregiver was handling multiple patients at the same real-world moment.
   MIMIC-III shifts each patient's dates by an independent random offset,
   so two patients' timestamps cannot be compared for simultaneity. This
   property is not computable and is left unpopulated. Correct the schema
   comment in the next OWL revision; do not attempt to fill this later
   without re-deriving a valid definition.

2. AcuityWeightedLoad is NOT populated in this step. It requires DailySOFA
   per ICU stay, computed separately via the MIT-LCP mimic-code SOFA
   queries. This is Step 3.

3. ActivityWindow is defined PER ICUSTAY, not as a shared calendar node
   across patients. This is required by the date-shifting constraint:
   only timestamps within one patient's own (shifted) clock are internally
   consistent. A window is (icustay_id, window_start); multiple caregivers
   can each have their own EventAggregate within the same window of the
   same stay.

4. precedes chains ActivityWindows within the SAME ICUStay only, ordered
   by window_start. It does not distinguish caregivers, since ActivityWindow
   itself is not caregiver-specific (this refines the earlier schema
   annotation, which described the constraint as "same caregiver AND same
   stay" -- with windows defined per-stay, "same stay" is the operative
   constraint; multiple caregivers sharing a window is expected).

5. InterWindowGap and ConsecutiveHighLoadCount are computed per
   (Caregiver, ICUStay) pair, i.e. within one patient encounter's clock,
   NOT across a caregiver's global shift pattern across all their patients.
   This is the same within-stay scope restriction as (3)/(4), applied
   consistently. State this as a limitation: these are continuity-of-care
   proxies within one encounter, not clinician-level shift recovery.
   ConsecutiveHighLoadCount uses a per-caregiver 75th-percentile threshold
   on EventCount, computed across ALL of that caregiver's windows in the
   full cohort (a valid statistical comparison; no cross-patient time
   alignment is implied by comparing two numbers).

6. EventCount = DocumentationIntensity + ProcedureCount, by construction:
     DocumentationIntensity = rows from CHARTEVENTS + DATETIMEEVENTS
     ProcedureCount         = rows from PROCEDUREEVENTS_MV
     EventCount              = the sum of both
   This resolves an ambiguity in the original property definitions, where
   both EventCount and DocumentationIntensity were described as counting
   CHARTEVENTS rows.

=============================================================================
"""
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, OWL, XSD

CCKG = Namespace("http://knacc.umbc.edu/ontologies/cckg#")

# MIMIC3_DIR = 
OUT_PATH = "cckg-instances-mimic-step2.ttl"

ADULT_ICUS = {"MICU", "CSRU", "SICU", "CCU", "TSICU"}
CHUNKSIZE = 2_000_000

# Set to a small integer (e.g. 3) for a fast test run over the first few
# chunks of each table before committing to the full pass. Set to None
# for the full run.
TEST_CHUNK_LIMIT = None


def find(name):
    for f in os.listdir(MIMIC3_DIR):
        if name.upper() in f.upper() and (f.endswith(".csv.gz") or f.endswith(".csv")):
            return os.path.join(MIMIC3_DIR, f)
    raise FileNotFoundError(name)


# ---------------------------------------------------------------------------
# Step 0: rebuild the same cohort filters used in populate_mimic_step1.py
# ---------------------------------------------------------------------------
print("Rebuilding cohort filters (must match Step 1 exactly)...")
icu = pd.read_csv(find("ICUSTAYS"))
icu.columns = [c.upper() for c in icu.columns]
icu = icu[icu["DBSOURCE"].str.lower() == "metavision"]
icu = icu[icu["FIRST_CAREUNIT"].isin(ADULT_ICUS)]
cohort_stays = set(icu["ICUSTAY_ID"].dropna().astype(int))
print(f"  Cohort ICU stays: {len(cohort_stays):,}")

cg = pd.read_csv(find("CAREGIVERS"))
cg.columns = [c.upper() for c in cg.columns]
cg = cg[cg["DESCRIPTION"].fillna("") != "Read Only"]
active_cgids = set(cg["CGID"].dropna().astype(int))
print(f"  Active caregivers: {len(active_cgids):,}")


# ---------------------------------------------------------------------------
# Step 1: windowing function (vectorized, 0600-1800 day / 1800-0600 night)
# ---------------------------------------------------------------------------
def compute_windows(ts):
    """Given a pandas Series of datetimes, return (window_start, window_type)."""
    date0 = ts.dt.normalize()
    hour = ts.dt.hour
    day_start = date0 + pd.Timedelta(hours=6)
    night_start_same = date0 + pd.Timedelta(hours=18)
    night_start_prev = date0 - pd.Timedelta(hours=6)

    mask_day = (hour >= 6) & (hour < 18)
    mask_late = hour >= 18
    # mask_early: hour < 6, handled by default assignment below

    window_start = night_start_prev.copy()
    window_start[mask_day] = day_start[mask_day]
    window_start[mask_late] = night_start_same[mask_late]

    window_type = np.where(mask_day, "day", "night")
    return window_start, window_type


# ---------------------------------------------------------------------------
# Step 2: chunked accumulation over the three event tables
# ---------------------------------------------------------------------------
def accumulate(file_key, timecol, counter):
    path = find(file_key)
    header = pd.read_csv(path, nrows=0)
    cols_upper = {c.upper(): c for c in header.columns}
    usecols = [cols_upper["ICUSTAY_ID"], cols_upper["CGID"], cols_upper[timecol]]

    print(f"Scanning {file_key} (usecols={usecols})...")
    rows_seen = 0
    rows_kept = 0
    n_chunks = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE):
        chunk.columns = [c.upper() for c in chunk.columns]
        rows_seen += len(chunk)

        chunk = chunk[chunk["ICUSTAY_ID"].notna() & chunk["CGID"].notna()]
        chunk["ICUSTAY_ID"] = chunk["ICUSTAY_ID"].astype(int)
        chunk["CGID"] = chunk["CGID"].astype(int)
        chunk = chunk[chunk["ICUSTAY_ID"].isin(cohort_stays) & chunk["CGID"].isin(active_cgids)]

        if len(chunk):
            ts = pd.to_datetime(chunk[timecol], errors="coerce")
            valid = ts.notna()
            chunk = chunk[valid]
            ts = ts[valid]
            wstart, wtype = compute_windows(ts)
            chunk = chunk.assign(WSTART=wstart, WTYPE=wtype)

            grp = chunk.groupby(["ICUSTAY_ID", "CGID", "WSTART", "WTYPE"]).size()
            for (stay, cgid, wstart_val, wtype_val), n in grp.items():
                counter[(stay, cgid, wstart_val, wtype_val)] += int(n)
            rows_kept += len(chunk)

        n_chunks += 1
        print(f"  chunk {n_chunks}: {rows_seen:,} rows scanned, "
              f"{rows_kept:,} kept so far, {len(counter):,} keys")

        if TEST_CHUNK_LIMIT is not None and n_chunks >= TEST_CHUNK_LIMIT:
            print(f"  TEST_CHUNK_LIMIT reached ({TEST_CHUNK_LIMIT}), stopping early.")
            break

    print(f"  {file_key} done: {rows_seen:,} scanned, {rows_kept:,} kept.\n")


doc_counts = defaultdict(int)   # CHARTEVENTS + DATETIMEEVENTS -> DocumentationIntensity
proc_counts = defaultdict(int)  # PROCEDUREEVENTS_MV -> ProcedureCount

accumulate("CHARTEVENTS", "CHARTTIME", doc_counts)
accumulate("DATETIMEEVENTS", "CHARTTIME", doc_counts)
accumulate("PROCEDUREEVENTS_MV", "STARTTIME", proc_counts)

# ---------------------------------------------------------------------------
# Step 3: merge into one table keyed by (stay, cgid, window_start, window_type)
# ---------------------------------------------------------------------------
print("Merging documentation and procedure counts...")
all_keys = set(doc_counts.keys()) | set(proc_counts.keys())
records = []
for key in all_keys:
    stay, cgid, wstart, wtype = key
    d = doc_counts.get(key, 0)
    p_ = proc_counts.get(key, 0)
    records.append({
        "ICUSTAY_ID": stay, "CGID": cgid, "WSTART": wstart, "WTYPE": wtype,
        "DocumentationIntensity": d, "ProcedureCount": p_, "EventCount": d + p_,
    })
ea_df = pd.DataFrame.from_records(records)
print(f"  {len(ea_df):,} (caregiver, stay, window) combinations")

# ---------------------------------------------------------------------------
# Step 4: recovery signals, per (Caregiver, ICUStay), within-stay only
# ---------------------------------------------------------------------------
print("Computing recovery signals (InterWindowGap, ConsecutiveHighLoadCount)...")

# Per-caregiver threshold: 75th percentile of EventCount across ALL their
# windows in the full cohort (a valid statistic; implies no cross-patient
# time comparison).
thresholds = ea_df.groupby("CGID")["EventCount"].quantile(0.75).to_dict()

ea_df["InterWindowGap"] = np.nan
ea_df["ConsecutiveHighLoadCount"] = 0

out_frames = []
for (stay, cgid), sub in ea_df.groupby(["ICUSTAY_ID", "CGID"]):
    sub = sub.sort_values("WSTART").copy()
    thr = thresholds.get(cgid, np.inf)

    gaps = [np.nan]
    prev_end = sub["WSTART"].iloc[0] + pd.Timedelta(hours=12)
    for i in range(1, len(sub)):
        this_start = sub["WSTART"].iloc[i]
        gap_hours = (this_start - prev_end).total_seconds() / 3600.0
        gaps.append(gap_hours)
        prev_end = this_start + pd.Timedelta(hours=12)
    sub["InterWindowGap"] = gaps

    streak = 0
    streaks = []
    for cnt in sub["EventCount"]:
        streak = streak + 1 if cnt > thr else 0
        streaks.append(streak)
    sub["ConsecutiveHighLoadCount"] = streaks

    out_frames.append(sub)

ea_df = pd.concat(out_frames, ignore_index=True)
print("  Recovery signals computed.\n")

# ---------------------------------------------------------------------------
# Step 5: build the RDF graph
# ---------------------------------------------------------------------------
print("Building RDF graph...")
g = Graph()
g.bind("cckg", CCKG)

# Distinct ActivityWindows: one per (ICUSTAY_ID, WSTART), reused across
# caregivers sharing that window.
window_keys = ea_df[["ICUSTAY_ID", "WSTART", "WTYPE"]].drop_duplicates()
window_uri = {}
for _, row in window_keys.iterrows():
    stay = int(row["ICUSTAY_ID"])
    wstart = row["WSTART"]
    key = (stay, wstart)
    wid = f"AW_{stay}_{wstart.strftime('%Y%m%dT%H%M')}"
    w = CCKG[wid]
    window_uri[key] = w
    g.add((w, RDF.type, OWL.NamedIndividual))
    g.add((w, RDF.type, CCKG.ActivityWindow))
    g.add((w, CCKG.WindowStart, Literal(wstart.isoformat(), datatype=XSD.dateTime)))
    g.add((w, CCKG.WindowEnd,
           Literal((wstart + pd.Timedelta(hours=12)).isoformat(), datatype=XSD.dateTime)))
    g.add((w, CCKG.WindowType, Literal(row["WTYPE"])))

print(f"  ActivityWindow individuals: {len(window_uri):,}")

# precedes: chain windows within the same stay, chronologically
n_precedes = 0
for stay, sub in window_keys.groupby("ICUSTAY_ID"):
    sub = sub.sort_values("WSTART")
    wstarts = list(sub["WSTART"])
    for i in range(len(wstarts) - 1):
        w1 = window_uri[(int(stay), wstarts[i])]
        w2 = window_uri[(int(stay), wstarts[i + 1])]
        g.add((w1, CCKG.precedes, w2))
        n_precedes += 1
print(f"  precedes edges: {n_precedes:,}")

# EventAggregate individuals
n_ea = 0
for _, row in ea_df.iterrows():
    stay = int(row["ICUSTAY_ID"])
    cgid = int(row["CGID"])
    wstart = row["WSTART"]
    eid = f"EA_{stay}_{cgid}_{wstart.strftime('%Y%m%dT%H%M')}"
    ea = CCKG[eid]
    g.add((ea, RDF.type, OWL.NamedIndividual))
    g.add((ea, RDF.type, CCKG.EventAggregate))
    g.add((ea, CCKG.EventCount, Literal(int(row["EventCount"]), datatype=XSD.int)))
    g.add((ea, CCKG.ProcedureCount, Literal(int(row["ProcedureCount"]), datatype=XSD.int)))
    g.add((ea, CCKG.DocumentationIntensity,
           Literal(int(row["DocumentationIntensity"]), datatype=XSD.decimal)))
    if pd.notna(row["InterWindowGap"]):
        g.add((ea, CCKG.InterWindowGap,
               Literal(round(float(row["InterWindowGap"]), 4), datatype=XSD.decimal)))
    g.add((ea, CCKG.ConsecutiveHighLoadCount,
           Literal(int(row["ConsecutiveHighLoadCount"]), datatype=XSD.int)))
    # UniqueStayCount deliberately NOT populated -- see header note (1).
    # AcuityWeightedLoad deliberately NOT populated -- see header note (2).

    stay_uri = CCKG[f"ICUStay_{stay}"]
    cg_uri = CCKG[f"Caregiver_{cgid}"]
    w_uri = window_uri[(stay, wstart)]

    g.add((ea, CCKG.involvesStay, stay_uri))
    g.add((ea, CCKG.belongsToWindow, w_uri))
    g.add((ea, CCKG.aggregateOf, cg_uri))
    g.add((cg_uri, CCKG.performsActivityIn, w_uri))
    n_ea += 1

print(f"  EventAggregate individuals: {n_ea:,}")

print("Serializing (this can take a few minutes for large graphs)...")
g.serialize(destination=OUT_PATH, format="turtle")
print(f"Wrote {len(g):,} triples -> {OUT_PATH}")
print()
print("Upload this file into the SAME 'cckg' TDB2 dataset in Fuseki,")
print("alongside cckg-schema.owl, cckg-instances-sops.ttl, and")
print("cckg-instances-mimic-step1.ttl. Then run the verification queries.")
