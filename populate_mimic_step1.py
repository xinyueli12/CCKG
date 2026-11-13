"""Populate the CCKG operational layer, Step 1: Caregivers, CareUnits,
Patients, ICUStays from local MIMIC-III files.

Run from the folder containing this script. Adjust MIMIC3_DIR below.
Output: cckg-instances-mimic-step1.ttl

Filters applied (all locked Phase 0/1 decisions):
  - ICUSTAYS restricted to DBSOURCE == 'metavision'
  - NICU stays excluded (adult ICUs only: MICU, CSRU, SICU, CCU, TSICU)
  - Caregivers with DESCRIPTION == 'Read Only' excluded

Deliberately deferred to Step 2 (recorded decisions):
  - ActivityWindow / EventAggregate: require the chunked CHARTEVENTS /
    PROCEDUREEVENTS_MV / DATETIMEEVENTS aggregation pipeline.
  - Restricting caregivers to those actually active in MetaVision events:
    requires scanning the event tables, so it lands with Step 2. Step 1
    loads all non-Read-Only caregivers; Step 2 can prune inactive ones.
  - performsActivityIn links: need ActivityWindows to exist first.
  - DailySOFA: computed via the MIT-LCP mimic-code queries in Step 2.

Expected runtime: 1-3 minutes. Expected output: roughly 400-700k triples
depending on cohort sizes after filtering.
"""
import os
import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, OWL, XSD

CCKG = Namespace("http://knacc.umbc.edu/ontologies/cckg#")

# MIMIC3_DIR = 
OUT_PATH = "cckg-instances-mimic-step1.ttl"

ADULT_ICUS = {"MICU", "CSRU", "SICU", "CCU", "TSICU"}

def find(name):
    for f in os.listdir(MIMIC3_DIR):
        if name.upper() in f.upper() and (f.endswith(".csv.gz") or f.endswith(".csv")):
            return os.path.join(MIMIC3_DIR, f)
    raise FileNotFoundError(name)

print("Loading ICUSTAYS...")
icu = pd.read_csv(find("ICUSTAYS"))
icu.columns = [c.upper() for c in icu.columns]
n_all = len(icu)
icu = icu[icu["DBSOURCE"].str.lower() == "metavision"]
icu = icu[icu["FIRST_CAREUNIT"].isin(ADULT_ICUS)]
print(f"  ICUSTAYS: {n_all:,} total -> {len(icu):,} after MetaVision + adult-ICU filters")

print("Loading CAREGIVERS...")
cg = pd.read_csv(find("CAREGIVERS"))
cg.columns = [c.upper() for c in cg.columns]
n_all = len(cg)
cg = cg[cg["DESCRIPTION"].fillna("") != "Read Only"]
print(f"  CAREGIVERS: {n_all:,} total -> {len(cg):,} after Read Only exclusion")

g = Graph()
g.bind("cckg", CCKG)

# --- CareUnits (5 individuals) ---
for unit in sorted(ADULT_ICUS):
    u = CCKG[f"CareUnit_{unit}"]
    g.add((u, RDF.type, OWL.NamedIndividual))
    g.add((u, RDF.type, CCKG.CareUnit))
    g.add((u, CCKG.UnitName, Literal(unit)))

# --- Caregivers ---
for _, row in cg.iterrows():
    c = CCKG[f"Caregiver_{int(row['CGID'])}"]
    g.add((c, RDF.type, OWL.NamedIndividual))
    g.add((c, RDF.type, CCKG.Caregiver))
    g.add((c, CCKG.CGID, Literal(int(row["CGID"]), datatype=XSD.int)))
    if pd.notna(row.get("DESCRIPTION")):
        g.add((c, CCKG.Role, Literal(str(row["DESCRIPTION"]))))
    if pd.notna(row.get("LABEL")):
        g.add((c, CCKG.RoleLabel, Literal(str(row["LABEL"]))))
print(f"  Caregiver individuals: {len(cg):,}")

# --- Patients (unique subjects in the filtered cohort) ---
subjects = icu["SUBJECT_ID"].dropna().astype(int).unique()
for sid in subjects:
    p = CCKG[f"Patient_{sid}"]
    g.add((p, RDF.type, OWL.NamedIndividual))
    g.add((p, RDF.type, CCKG.Patient))
    g.add((p, CCKG.SubjectID, Literal(int(sid), datatype=XSD.int)))
print(f"  Patient individuals: {len(subjects):,}")

# --- ICUStays with occursInUnit and hasStay links ---
n_stays = 0
for _, row in icu.iterrows():
    sid = int(row["ICUSTAY_ID"])
    s = CCKG[f"ICUStay_{sid}"]
    g.add((s, RDF.type, OWL.NamedIndividual))
    g.add((s, RDF.type, CCKG.ICUStay))
    g.add((s, CCKG.StayID, Literal(sid, datatype=XSD.int)))
    if pd.notna(row.get("INTIME")):
        g.add((s, CCKG.InTime,
               Literal(str(row["INTIME"]).replace(" ", "T"), datatype=XSD.dateTime)))
    if pd.notna(row.get("OUTTIME")):
        g.add((s, CCKG.OutTime,
               Literal(str(row["OUTTIME"]).replace(" ", "T"), datatype=XSD.dateTime)))
    if pd.notna(row.get("LOS")):
        g.add((s, CCKG.LengthOfStay,
               Literal(round(float(row["LOS"]), 4), datatype=XSD.decimal)))
    g.add((s, CCKG.occursInUnit, CCKG[f"CareUnit_{row['FIRST_CAREUNIT']}"]))
    g.add((CCKG[f"Patient_{int(row['SUBJECT_ID'])}"], CCKG.hasStay, s))
    n_stays += 1
print(f"  ICUStay individuals: {n_stays:,}")

print("Serializing (this is the slow part, be patient)...")
g.serialize(destination=OUT_PATH, format="turtle")
print(f"Wrote {len(g):,} triples -> {OUT_PATH}")
print()
print("Next: create a PERSISTENT (TDB2) dataset in Fuseki named e.g. 'cckg',")
print("upload cckg-schema.owl, cckg-instances-sops.ttl, and this file into it,")
print("then run the verification queries.")
