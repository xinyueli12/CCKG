"""Populate the CCKG construct layer (SOPS side) from the WPS hospital table.

Reads wps_composites_hospital_level.csv (38 hospitals, computed earlier in the
project) and emits cckg-instances-sops.ttl containing:
  - one HospitalContext individual per hospital (5 schema-defined properties)
  - one SurveyEvent individual per hospital (InstrumentVersion = WPS)
  - hasSurveyEvent links

Deliberately NOT populated (recorded decisions, not omissions):
  - The six WPS composite scores: no schema property exists for per-wave
    observed composite values; that modeling is deferred pending the SOPS
    vintage decision (v4 diff). Populating them would require inventing
    properties outside the schema, which breaks the discipline.
  - SurveyQuarter: the hospital-level CSV carries no per-hospital
    administration date; fabricating a quarter from the collection window
    would be a guess. Omitted until a date source is confirmed.
  - GapMonths: single wave, no predecessor; null by definition.
  - temporalNext: single wave, nothing to chain.
"""
import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef, RDF, OWL, XSD

CCKG = Namespace("http://knacc.umbc.edu/ontologies/cckg#")

#CSV_PATH = 
OUT_PATH = "cckg-instances-sops.ttl"

df = pd.read_csv(CSV_PATH)

g = Graph()
g.bind("cckg", CCKG)

n_hosp, n_evt = 0, 0
for _, row in df.iterrows():
    pid = int(row["PUBLICID"])

    hc = CCKG[f"HospitalContext_{pid}"]
    g.add((hc, RDF.type, OWL.NamedIndividual))
    g.add((hc, RDF.type, CCKG.HospitalContext))
    g.add((hc, CCKG.PublicID, Literal(pid, datatype=XSD.int)))
    g.add((hc, CCKG.SurveyWave, Literal("2021/22 WPS")))
    if pd.notna(row["burnout_pct_3plus"]):
        g.add((hc, CCKG.BurnoutPct3Plus,
               Literal(round(float(row["burnout_pct_3plus"]), 4), datatype=XSD.decimal)))
    if pd.notna(row["leave_pct_any_yes"]):
        g.add((hc, CCKG.LeavePctAnyYes,
               Literal(round(float(row["leave_pct_any_yes"]), 4), datatype=XSD.decimal)))
    if pd.notna(row["RESPONSERATE"]):
        g.add((hc, CCKG.ResponseRate,
               Literal(round(float(row["RESPONSERATE"]), 4), datatype=XSD.decimal)))
    n_hosp += 1

    ev = CCKG[f"SurveyEvent_WPS_{pid}"]
    g.add((ev, RDF.type, OWL.NamedIndividual))
    g.add((ev, RDF.type, CCKG.SurveyEvent))
    g.add((ev, CCKG.InstrumentVersion, Literal("WPS")))
    g.add((hc, CCKG.hasSurveyEvent, ev))
    n_evt += 1

g.serialize(destination=OUT_PATH, format="turtle")
print(f"Wrote {len(g)} triples: {n_hosp} HospitalContext, {n_evt} SurveyEvent -> {OUT_PATH}")
