"""
CCKG Phase 2, Step 4 completion: attach per-hospital-per-wave SOPS composite
values to the graph, and add the schema elements required to hold them.

Output: cckg-instances-sops-constructs-v4.ttl
"""
import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, OWL, RDFS, XSD

CCKG = Namespace("http://knacc.umbc.edu/ontologies/cckg#")

#WPS_CSV = 
#H20_CSV = 
OUT = "cckg-instances-sops-constructs.ttl"

# --- construct registry -------------------------------------------------
# v4: POLARITY added as the 8th field. "higher_better" means a larger value is
# a more favourable state; "higher_worse" means a larger value is a worse state.
# NOT derivable from JDRClassification: BurnoutWPG1, IntentToLeaveWPS and
# JobSatisfactionWPS are all "outcome", but the first two are higher_worse and
# the third higher_better. NOT derivable from the property name either:
# PercentPositive is a misnomer for the two higher_worse constructs, whose
# source columns are burnout_pct_3plus and leave_pct_any_yes (percent reporting
# a problem). Verified empirically: Burnout vs IntentToLeave rho=+0.648 (same
# polarity); Burnout vs JobSatisfaction rho=-0.766 (opposite). Any computation
# combining these values -- notably ResourceAdequacy in the well-being-risk
# score -- must orient on this field, not on the numeric scale.
# (construct individual, source column, instrument, JD-R class, item count,
#  measurement type, scale)
# ItemCount values marked VERIFY are carried from the existing schema file and
# are NOT re-derived here; the two corrected counts (Management Support 3,
# Handoffs 3) are already applied in cckg-schema.owl.
CONSTRUCTS = [
    # HSOPS 2.0 core composites
    ("StaffingAndWorkPace",           "POS_STAFF",   "HSOPS2", "demand",   4, "composite", "prop", "higher_better"),
    ("HandoffsInformationExchange",   "POS_HANDOFF", "HSOPS2", "resource", 3, "composite", "prop", "higher_better"),
    ("ManagementSupportPatientSafety","POS_MGMT",    "HSOPS2", "resource", 3, "composite", "prop", "higher_better"),
    ("CommunicationOpenness",         "POS_COMMUN",  "HSOPS2", "resource", 4, "composite", "prop", "higher_better"),
    ("TeamworkWithinUnits",           "POS_TEAM",    "HSOPS2", "resource", 3, "composite", "prop", "higher_better"),
    ("SupervisorSupportPatientSafety","POS_SUPV",    "HSOPS2", "resource", 3, "composite", "prop", "higher_better"),
    # WPS supplement composites
    ("ProtectionFromWorkplaceHazards","Protection_from_Workplace_Hazards",    "WPS", "resource", 4, "composite", "prop", "higher_better"),
    ("MovingTransferringLifting",     "Moving_Transferring_Lifting_Patients", "WPS", "resource", 3, "composite", "prop", "higher_better"),
    ("WorkplaceAggression",           "Workplace_Aggression",                 "WPS", "demand",   4, "composite", "prop", "higher_better"),
    ("SupervisorSupportWorkplaceSafety","Supervisor_Support_Workplace_Safety","WPS", "resource", 3, "composite", "prop", "higher_better"),
    ("ManagementSupportWorkplaceSafety","Management_Support_Workplace_Safety","WPS", "resource", 3, "composite", "prop", "higher_better"),
    ("ReportingWorkplaceSafetyConcerns","Reporting_Workplace_Safety_Concerns","WPS", "resource", 3, "composite", "prop", "higher_better"),
    # WPS single-item outcomes
    ("BurnoutWPG1",                   "burnout_pct_3plus",        "WPS", "outcome", 1, "single-item", "pct", "higher_worse"),
    ("IntentToLeaveWPS",              "leave_pct_any_yes",        "WPS", "outcome", 1, "single-item", "pct", "higher_worse"),
    ("JobSatisfactionWPS",            "satisfaction_pct_positive","WPS", "outcome", 1, "single-item", "pct", "higher_better"),
]

INSTRUMENT_META = {
    "HSOPS2": ("HSOPS 2.0", "2022", "hsops2_db2022_hospital.csv"),
    "WPS":    ("WPS",       "2021/22 WPS", "wps_composites_hospital_level.csv"),
}


def main():
    wps = pd.read_csv(WPS_CSV)
    h20 = pd.read_csv(H20_CSV)
    wps["PUBLICID"] = wps.PUBLICID.astype(int)
    h20["PUBLICID"] = h20.PUBLICID.astype(int)

    # restrict HSOPS 2.0 to the 38 hospitals that also have WPS data, so the
    # construct layer covers one consistent hospital set.
    ids = sorted(set(wps.PUBLICID) & set(h20.PUBLICID))
    assert len(ids) == 38, f"expected 38 joined hospitals, got {len(ids)}"
    h20 = h20[h20.PUBLICID.isin(ids)].set_index("PUBLICID")
    wps = wps[wps.PUBLICID.isin(ids)].set_index("PUBLICID")
    src = {"HSOPS2": h20, "WPS": wps}

    g = Graph()
    g.bind("cckg", CCKG)
    g.bind("owl", OWL)

    # ---- schema additions ----------------------------------------------
    g.add((CCKG.ConstructMeasurement, RDF.type, OWL.Class))
    g.add((CCKG.ConstructMeasurement, RDFS.comment, Literal(
        "One SOPS composite or single-item value for one hospital in one "
        "survey administration. Reified because the value is a property of "
        "the (SurveyEvent, SurveyConstruct) pair, not of either alone.")))

    for prop, dom, rng, cmt in [
        ("hasConstructMeasurement", CCKG.SurveyEvent, CCKG.ConstructMeasurement,
         "Links a survey administration to the values it reported."),
        ("measuresConstruct", CCKG.ConstructMeasurement, CCKG.SurveyConstruct,
         "Links a reported value to the construct it measures."),
    ]:
        g.add((CCKG[prop], RDF.type, OWL.ObjectProperty))
        g.add((CCKG[prop], RDFS.domain, dom))
        g.add((CCKG[prop], RDFS.range, rng))
        g.add((CCKG[prop], RDFS.comment, Literal(cmt)))

    for prop, rng, cmt in [
        ("PercentPositive", XSD.decimal,
         "Percent-positive response, normalised to 0-100 across instruments."),
        ("ValueScale", XSD.string,
         "Native published scale of the source column: 'percent' or 'proportion'."),
        ("SourceColumn", XSD.string, "Source file column name, for provenance."),
        ("SourceFile", XSD.string, "Source file name, for provenance."),
        ("Polarity", XSD.string,
         "higher_better | higher_worse. Orientation of the value. Required "
         "because PercentPositive is a misnomer for burnout and intent-to-leave, "
         "and because JDRClassification does not determine polarity."),
    ]:
        g.add((CCKG[prop], RDF.type, OWL.DatatypeProperty))
        g.add((CCKG[prop], RDFS.domain, CCKG.ConstructMeasurement))
        g.add((CCKG[prop], RDFS.range, rng))
        g.add((CCKG[prop], RDFS.comment, Literal(cmt)))

    # ---- construct individuals -----------------------------------------
    # v2 FIX: 5 of these 15 names already exist as SurveyConstruct individuals
    # in the schema with a spaced ConstructName ("Staffing and Work Pace") and
    # their own ItemCount/JDRClassification/MeasurementType. v1 of this script
    # re-asserted a SECOND ConstructName (code-style, no spaces) on the SAME
    # URI, which does not overwrite in RDF -- it adds a second value, and every
    # downstream query grouping/joining on ConstructName doubles. Fix: skip
    # identity metadata for names already declared in the schema; only their
    # ConstructMeasurement facts get populated below.
    PRE_EXISTING_IN_SCHEMA = {
        "StaffingAndWorkPace", "HandoffsInformationExchange",
        "ManagementSupportPatientSafety", "CommunicationOpenness", "BurnoutWPG1",
    }
    seen = set()
    for name, col, inst, jdr, items, mtype, _scale, _pol in CONSTRUCTS:
        if name in seen:
            continue
        seen.add(name)
        u = CCKG[name]
        if name in PRE_EXISTING_IN_SCHEMA:
            continue  # identity already asserted once, in the schema file
        g.add((u, RDF.type, CCKG.SurveyConstruct))
        g.add((u, RDF.type, OWL.NamedIndividual))
        g.add((u, CCKG.ConstructName, Literal(name)))
        g.add((u, CCKG.ItemCount, Literal(items, datatype=XSD.int)))
        g.add((u, CCKG.JDRClassification, Literal(jdr)))
        g.add((u, CCKG.MeasurementType, Literal(mtype)))

    # ---- survey events + measurements ----------------------------------
    n_meas = 0
    for hid in ids:
        hosp = CCKG[f"HospitalContext_{hid}"]
        for inst, (iver, wave, fname) in INSTRUMENT_META.items():
            ev = CCKG[f"SurveyEvent_{inst}_{hid}"]
            g.add((ev, RDF.type, CCKG.SurveyEvent))
            g.add((ev, RDF.type, OWL.NamedIndividual))
            g.add((ev, CCKG.InstrumentVersion, Literal(iver)))
            g.add((ev, CCKG.SurveyWave, Literal(wave)))
            g.add((hosp, CCKG.hasSurveyEvent, ev))

            for name, col, cinst, _j, _i, _m, scale, polarity in CONSTRUCTS:
                if cinst != inst:
                    continue
                val = src[inst].at[hid, col]
                if pd.isna(val):
                    continue  # missing values are omitted, not imputed
                pct = float(val) * 100.0 if scale == "prop" else float(val)
                m = CCKG[f"CM_{inst}_{hid}_{name}"]
                g.add((m, RDF.type, CCKG.ConstructMeasurement))
                g.add((m, RDF.type, OWL.NamedIndividual))
                g.add((ev, CCKG.hasConstructMeasurement, m))
                g.add((m, CCKG.measuresConstruct, CCKG[name]))
                g.add((m, CCKG.PercentPositive,
                       Literal(round(pct, 4), datatype=XSD.decimal)))
                g.add((m, CCKG.ValueScale,
                       Literal("proportion" if scale == "prop" else "percent")))
                g.add((m, CCKG.SourceColumn, Literal(col)))
                g.add((m, CCKG.SourceFile, Literal(fname)))
                g.add((m, CCKG.Polarity, Literal(polarity)))
                n_meas += 1

    g.serialize(OUT, format="turtle")
    print(f"hospitals: {len(ids)}")
    print(f"constructs: {len(seen)}")
    print(f"measurements: {n_meas}")
    print(f"triples: {len(g)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
