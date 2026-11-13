"""Generate the CCKG Phase 1 schema as OWL/RDF-XML."""
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD

CCKG = Namespace("http://knacc.umbc.edu/ontologies/cckg#")
ONT_IRI = URIRef("http://knacc.umbc.edu/ontologies/cckg")

g = Graph()
g.bind("cckg", CCKG)
g.bind("owl", OWL)

# ---------------------------------------------------------------- v4 flags
# ClinicalActivity is declared in the schema but has no instances in any
# populated file: event-level materialization was deferred on tractability
# grounds (the three MetaVision event tables are ~10^8 rows). Set this False
# to drop the class and its 4 data properties / 3 object properties entirely.
# Whichever value is used, Section III-D must state the choice explicitly:
# a declared-but-empty class is defensible as an extension point, but only
# if named as one.
MATERIALIZE_CLINICAL_ACTIVITY = True

g.add((ONT_IRI, RDF.type, OWL.Ontology))
g.add((ONT_IRI, RDFS.comment, Literal(
    "Clinician-Centered Knowledge Graph (CCKG) schema, Phase 1. "
    "Caregiver-recentered ontology for ICU workforce analytics integrating "
    "MIMIC-III operational activity with AHRQ SOPS normative constructs. "
    "Structural patterns follow Walid et al. (2021, 2024): flat class "
    "hierarchy, PascalCase data properties, camelCase object properties, "
    "100% domain/range specification. Ontology IRI is provisional pending "
    "lab confirmation.")))

# ---------------------------------------------------------------- classes
classes = {
    "Caregiver": "Instance source: MIMIC-III CAREGIVERS (CGID, LABEL, DESCRIPTION). "
                 "Exclude DESCRIPTION='Read Only'. MetaVision-active caregivers only.",
    "CareUnit": "Instance source: MIMIC-III ICUSTAYS.FIRST_CAREUNIT. Adult ICUs only: "
                "MICU, CSRU, SICU, CCU, TSICU. NICU excluded.",
    "Patient": "Instance source: MIMIC-III ICUSTAYS.SUBJECT_ID. Retained but secondary "
               "to Caregiver (entity recentering).",
    "ICUStay": "Instance source: MIMIC-III ICUSTAYS filtered to DBSOURCE='metavision'.",
    "ActivityWindow": "Derived: 12-hour windows aligned 0600-1800 (day) / 1800-0600 "
                      "(night), from CHARTTIME. Valid only within a single caregiver's "
                      "sequence on a single stay's clock; MIMIC-III per-patient date "
                      "shifting forbids cross-patient temporal comparison.",
    "EventAggregate": "Derived: per-caregiver per-window event summary from CHARTEVENTS, "
                      "PROCEDUREEVENTS_MV, DATETIMEEVENTS. Workload and recovery proxies "
                      "live here.",
    "ClinicalActivity": "DECLARED, NOT POPULATED at v4 (see MATERIALIZE_CLINICAL_ACTIVITY). One documented clinical activity (event-level task instance), "
                        "following the agent-executes-task pattern of industrial process "
                        "ontologies (IOF, referenced as architecture, not imported). "
                        "Instance source: single rows of CHARTEVENTS, PROCEDUREEVENTS_MV, "
                        "or DATETIMEEVENTS. Materialized for a cohort subsample only; "
                        "full-scale features remain on EventAggregate. Answers the "
                        "time-series representation question: each activity is a "
                        "first-class node carrying its timestamp, windows partition "
                        "activities, precedes orders windows.",
    "SurveyConstruct": "Instance source: AHRQ SOPS Hospital Survey 2.0 composites and "
                       "WPS Supplement composites/items. MeasurementType distinguishes "
                       "validated multi-item composites from single-item measures "
                       "(WPG1, WPBQ1, WPBQ2, WPF1).",
    "HospitalContext": "Instance source: SOPS hospital-level files (wps_composites_"
                       "hospital_level.csv; N=38 usable WPS hospitals). Normative "
                       "reference frame per survey wave. Deliberately NOT temporally "
                       "linked to ActivityWindow: SOPS priors are contextual reference "
                       "values, not synchronized observations.",
    "SurveyEvent": "One SOPS survey administration for one hospital (reified wave). "
                   "Pattern adapted from Zeliatu's trajectory ontology: reifying each "
                   "administration as a first-class temporal node enables hospital-level "
                   "wellness trajectories via temporalNext ordering. Instance source: "
                   "SOPS hospital files, one event per database submission "
                   "(SURVDATE / STARTMONTH-ENDMONTH derive the quarter). "
                   "InstrumentVersion is mandatory at population time: AHRQ states "
                   "HSOPS 1.0 and 2.0 scores are not comparable, so trajectories are "
                   "valid only within a single instrument version. Attachment of "
                   "per-wave observed composite values is deliberately deferred to "
                   "population design, pending the SOPS vintage decision.",
    "WellbeingRiskAssessment": "Derived: one computed risk score per caregiver-window "
                               "with its inputs retained for auditability. Formula: "
                               "demand x (1 - resource adequacy) x (1 - recovery "
                               "adequacy). MLLO MLTask target (Level 1 mapping row 5).",
    "ProxyMapping": "Reified bridge edge. Carries the assumption, provenance type "
                    "(empirical/normative/engineered), and pre-registered predicted "
                    "direction for each operational proxy -> SOPS construct mapping. "
                    "PredictedDirection is written before graph population; the "
                    "construct-alignment audit compares it against observed direction.",
}
for name, comment in classes.items():
    c = CCKG[name]
    g.add((c, RDF.type, OWL.Class))
    g.add((c, RDFS.subClassOf, OWL.Thing))
    g.add((c, RDFS.comment, Literal(comment)))

# --------------------------------------------------------- object properties
obj_props = [
    ("performsActivityIn", "Caregiver", "ActivityWindow"),
    ("hasAggregate", "ActivityWindow", "EventAggregate"),
    ("aggregateOf", "EventAggregate", "Caregiver"),
    ("involvesStay", "EventAggregate", "ICUStay"),
    ("occursInUnit", "ICUStay", "CareUnit"),
    ("hasStay", "Patient", "ICUStay"),
    ("precedes", "ActivityWindow", "ActivityWindow"),
    ("assessedIn", "WellbeingRiskAssessment", "ActivityWindow"),
    ("assessmentOf", "WellbeingRiskAssessment", "Caregiver"),
    ("contextualizedBy", "HospitalContext", "SurveyConstruct"),
    ("mapsToConstruct", "ProxyMapping", "SurveyConstruct"),
    ("executedBy", "ClinicalActivity", "Caregiver"),
    ("partOfStay", "ClinicalActivity", "ICUStay"),
    # v4 CORRECTION: domain was ClinicalActivity, which is a defect, not a
    # design choice. populate_mimic_step2.py emits belongsToWindow on
    # EventAggregate individuals, and the caregiver-timeline query reads it
    # from EventAggregate. Under OWL semantics the old declaration made a
    # reasoner infer EventAggregate rdfs:subClassOf ClinicalActivity for every
    # populated aggregate. Domain corrected to the class that actually uses it.
    ("belongsToWindow", "EventAggregate", "ActivityWindow"),
    ("hasSurveyEvent", "HospitalContext", "SurveyEvent"),
    ("temporalNext", "SurveyEvent", "SurveyEvent"),
]
for name, dom, rng in obj_props:
    p = CCKG[name]
    g.add((p, RDF.type, OWL.ObjectProperty))
    g.add((p, RDFS.domain, CCKG[dom]))
    g.add((p, RDFS.range, CCKG[rng]))

pc = CCKG["precedes"]
g.add((pc, RDFS.comment, Literal(
    "Temporal spine for time-series feature extraction. Constraint (enforced in "
    "population pipeline, not OWL): only between consecutive windows of the same "
    "caregiver within one stay's clock.")))

tn = CCKG["temporalNext"]
g.add((tn, RDFS.comment, Literal(
    "Hospital-level trajectory spine (pattern adapted from Zeliatu's trajectory "
    "ontology). Constraint (enforced in population pipeline, not OWL): only "
    "between consecutive SurveyEvents of the same hospital AND the same "
    "InstrumentVersion. AHRQ HSOPS 1.0 and 2.0 scores are not comparable; "
    "cross-version chains produce instrument artifacts, not trajectories. "
    "Auditable via the InstrumentVersion property on both linked events.")))

# ----------------------------------------------------------- data properties
dt = {"int": XSD.int, "string": XSD.string, "decimal": XSD.decimal,
      "dateTime": XSD.dateTime}
data_props = [
    # Caregiver
    ("CGID", "Caregiver", "int", "MIMIC-III CAREGIVERS.CGID"),
    ("Role", "Caregiver", "string", "MIMIC-III CAREGIVERS.DESCRIPTION (16 categories)"),
    ("RoleLabel", "Caregiver", "string", "MIMIC-III CAREGIVERS.LABEL (230 codes, traceability)"),
    # CareUnit
    ("UnitName", "CareUnit", "string", "MIMIC-III ICUSTAYS.FIRST_CAREUNIT"),
    # Patient
    ("SubjectID", "Patient", "int", "MIMIC-III ICUSTAYS.SUBJECT_ID"),
    # ICUStay
    ("StayID", "ICUStay", "int", "MIMIC-III ICUSTAYS.ICUSTAY_ID"),
    ("InTime", "ICUStay", "dateTime", "MIMIC-III ICUSTAYS.INTIME"),
    ("OutTime", "ICUStay", "dateTime", "MIMIC-III ICUSTAYS.OUTTIME"),
    ("LengthOfStay", "ICUStay", "decimal", "MIMIC-III ICUSTAYS.LOS"),
    ("DailySOFA", "ICUStay", "decimal", "Computed via MIT-LCP/mimic-code SOFA script"),
    # ActivityWindow
    ("WindowStart", "ActivityWindow", "dateTime", "Derived from CHARTTIME"),
    ("WindowEnd", "ActivityWindow", "dateTime", "Derived from CHARTTIME"),
    ("WindowType", "ActivityWindow", "string", "day | night (0600/1800 boundaries)"),
    # EventAggregate
    ("EventCount", "EventAggregate", "int", "Count of CHARTEVENTS rows per CGID-window"),
    ("ProcedureCount", "EventAggregate", "int", "Count of PROCEDUREEVENTS_MV rows per CGID-window"),
    ("UniqueStayCount", "EventAggregate", "int",
     "DECLARED, NOT POPULATED (v4 correction). The original comment read "
     "'Distinct ICUSTAY_ID per CGID-window', which is not computable and, at "
     "this domain, not informative. Two independent reasons. (a) The "
     "pre-registered Assumption on PM_UniqueStays_Handoffs is about CONCURRENT "
     "stays; establishing simultaneity requires comparing timestamps across "
     "patients, which MIMIC-III per-patient random date shifting forbids. "
     "(b) EventAggregate is scoped to exactly one ICUStay by construction "
     "(involvesStay is functional in practice), so distinct ICUSTAY_ID per "
     "EventAggregate is 1 for every instance: the property carries no variance "
     "and supports no correlation even under perfect calendar alignment. "
     "Retained in the schema so the pre-registered mapping remains auditable; "
     "reported as untestable-by-construction. Do not redefine post hoc."),
    ("AcuityWeightedLoad", "EventAggregate", "decimal", "EventCount weighted by stay SOFA"),
    ("DocumentationIntensity", "EventAggregate", "decimal", "Chart events per window"),
    ("InterWindowGap", "EventAggregate", "decimal", "Hours between consecutive active windows (recovery signal)"),
    ("ConsecutiveHighLoadCount", "EventAggregate", "int",
     "Consecutive windows above per-caregiver top-quartile threshold. "
     "Threshold is an engineered decision; see linked ProxyMapping."),
    # ClinicalActivity
    ("ActivityType", "ClinicalActivity", "string",
     "Event category; from source table and D_ITEMS label of the ITEMID"),
    ("ActivityTimestamp", "ClinicalActivity", "dateTime",
     "CHARTTIME (or STARTTIME for procedures); valid within one stay's shifted clock"),
    ("SourceTable", "ClinicalActivity", "string",
     "CHARTEVENTS | PROCEDUREEVENTS_MV | DATETIMEEVENTS (row-level provenance)"),
    ("SourceItemID", "ClinicalActivity", "int", "MIMIC-III ITEMID of the documented event"),
    # SurveyConstruct
    ("ConstructName", "SurveyConstruct", "string", "SOPS composite/item name"),
    ("ItemCount", "SurveyConstruct", "int", "Items in composite; 1 for single-item measures"),
    ("PercentPositive", "SurveyConstruct", "decimal",
     "DEPRECATED in v4. A single slot on the construct cannot hold 38 "
     "hospital-specific values; per-hospital-per-wave values now live on "
     "ConstructMeasurement, reified over the (SurveyEvent, SurveyConstruct) "
     "pair. Retained only so existing queries do not break; do not populate."),
    ("JDRClassification", "SurveyConstruct", "string", "demand | resource | outcome (JD-R theory)"),
    ("MeasurementType", "SurveyConstruct", "string",
     "composite | single-item. Single items (WPG1, WPBQ1, WPBQ2, WPF1) have no "
     "Cronbach's alpha or CFA; validity rests on AHRQ survey development."),
    # HospitalContext
    ("PublicID", "HospitalContext", "int", "SOPS PUBLICID"),
    ("SurveyWave", "HospitalContext", "string", "e.g., 2021/22 WPS; 2022 Hospital 2.0"),
    ("BurnoutPct3Plus", "HospitalContext", "decimal", "Pct respondents scoring 3+ on WPG1 (computed from individual file)"),
    ("LeavePctAnyYes", "HospitalContext", "decimal", "Pct WPBQ2 != No (computed from individual file)"),
    ("ResponseRate", "HospitalContext", "decimal", "SOPS RESPONSERATE"),
    # SurveyEvent
    ("SurveyQuarter", "SurveyEvent", "string",
     "Calendar quarter of administration completion, e.g. 2022Q2; derived from "
     "SURVDATE or STARTMONTH/ENDMONTH"),
    ("GapMonths", "SurveyEvent", "decimal",
     "Months elapsed since this hospital's previous SurveyEvent of the same "
     "InstrumentVersion; null for the first event in a chain"),
    ("InstrumentVersion", "SurveyEvent", "string",
     "HSOPS-1.0 | HSOPS-2.0 | WPS. Mandatory: makes the temporalNext "
     "same-version constraint auditable. SurveyQuarter cannot substitute; the "
     "1.0 and 2.0 collection windows overlap (Dec 2017 - Oct 2020 vs "
     "Nov 2018 - Oct 2020)"),
    # WellbeingRiskAssessment
    ("RiskScore", "WellbeingRiskAssessment", "decimal", "demand x (1-resource) x (1-recovery)"),
    ("DemandInput", "WellbeingRiskAssessment", "decimal", "Normalized demand component"),
    ("ResourceInput", "WellbeingRiskAssessment", "decimal", "Normalized resource adequacy component"),
    ("RecoveryInput", "WellbeingRiskAssessment", "decimal", "Normalized recovery adequacy component"),
    ("FormulaVersion", "WellbeingRiskAssessment", "string", "Version tag; formula will be revised after team scope meeting"),
    # ProxyMapping
    ("ProxyPropertyName", "ProxyMapping", "string", "Name of the EventAggregate data property being mapped"),
    ("Assumption", "ProxyMapping", "string", "The claim the mapping rests on, stated explicitly"),
    ("ProvenanceType", "ProxyMapping", "string", "empirical | normative | engineered"),
    ("PredictedDirection", "ProxyMapping", "string",
     "positive | negative. Sign of predicted association between proxy value and "
     "the construct's stated pole. Pre-registered before population."),
]
for name, dom, rng, comment in data_props:
    p = CCKG[name]
    g.add((p, RDF.type, OWL.DatatypeProperty))
    g.add((p, RDFS.domain, CCKG[dom]))
    g.add((p, RDFS.range, dt[rng]))
    g.add((p, RDFS.comment, Literal("Source: " + comment)))

# ---------------------------------------------------------------- individuals
def indiv(name, cls, data=None, obj=None):
    i = CCKG[name]
    g.add((i, RDF.type, OWL.NamedIndividual))
    g.add((i, RDF.type, CCKG[cls]))
    for prop, val, typ in (data or []):
        g.add((i, CCKG[prop], Literal(val, datatype=dt[typ])))
    for prop, target in (obj or []):
        g.add((i, CCKG[prop], CCKG[target]))

# SurveyConstruct individuals (verified against codebooks / WPS audit)
indiv("StaffingAndWorkPace", "SurveyConstruct", data=[
    ("ConstructName", "Staffing and Work Pace", "string"),
    ("ItemCount", 4, "int"),
    ("JDRClassification", "resource", "string"),
    ("MeasurementType", "composite", "string")])
indiv("CommunicationOpenness", "SurveyConstruct", data=[
    ("ConstructName", "Communication Openness", "string"),
    ("ItemCount", 4, "int"),
    ("JDRClassification", "resource", "string"),
    ("MeasurementType", "composite", "string")])
indiv("HandoffsInformationExchange", "SurveyConstruct", data=[
    ("ConstructName", "Handoffs and Information Exchange", "string"),
    ("ItemCount", 3, "int"),
    ("JDRClassification", "resource", "string"),
    ("MeasurementType", "composite", "string")])
indiv("ManagementSupportPatientSafety", "SurveyConstruct", data=[
    ("ConstructName", "Hospital Management Support for Patient Safety", "string"),
    ("ItemCount", 3, "int"),
    ("JDRClassification", "resource", "string"),
    ("MeasurementType", "composite", "string")])
indiv("BurnoutWPG1", "SurveyConstruct", data=[
    ("ConstructName", "Work Stress/Burnout (WPG1)", "string"),
    ("ItemCount", 1, "int"),
    ("JDRClassification", "outcome", "string"),
    ("MeasurementType", "single-item", "string")])

# ProxyMapping individuals (pre-registered predictions)
indiv("PM_DocIntensity_Staffing", "ProxyMapping",
      data=[("ProxyPropertyName", "DocumentationIntensity", "string"),
            ("Assumption", "Documentation volume per window indexes task demand; "
             "higher intensity predicts lower hospital staffing percent-positive.", "string"),
            ("ProvenanceType", "engineered", "string"),
            ("PredictedDirection", "negative", "string")],
      obj=[("mapsToConstruct", "StaffingAndWorkPace")])
indiv("PM_AcuityLoad_Staffing", "ProxyMapping",
      data=[("ProxyPropertyName", "AcuityWeightedLoad", "string"),
            ("Assumption", "SOFA-weighted event load indexes demand intensity; "
             "higher load predicts lower staffing percent-positive.", "string"),
            ("ProvenanceType", "engineered", "string"),
            ("PredictedDirection", "negative", "string")],
      obj=[("mapsToConstruct", "StaffingAndWorkPace")])
indiv("PM_InterWindowGap_Burnout", "ProxyMapping",
      data=[("ProxyPropertyName", "InterWindowGap", "string"),
            ("Assumption", "Effort-Recovery: shorter inter-window gaps mean "
             "insufficient recovery; gap length predicts burnout severity "
             "negatively (longer gap, lower burnout).", "string"),
            ("ProvenanceType", "normative", "string"),
            ("PredictedDirection", "negative", "string")],
      obj=[("mapsToConstruct", "BurnoutWPG1")])
indiv("PM_ConsecHighLoad_Burnout", "ProxyMapping",
      data=[("ProxyPropertyName", "ConsecutiveHighLoadCount", "string"),
            ("Assumption", "JD-R plus Effort-Recovery: sustained high-demand "
             "sequences without recovery amplify strain; count predicts burnout "
             "severity positively.", "string"),
            ("ProvenanceType", "normative", "string"),
            ("PredictedDirection", "positive", "string")],
      obj=[("mapsToConstruct", "BurnoutWPG1")])
indiv("PM_UniqueStays_Handoffs", "ProxyMapping",
      data=[("ProxyPropertyName", "UniqueStayCount", "string"),
            ("Assumption", "More concurrent stays per window means more information "
             "exchange surfaces; higher count predicts lower handoffs "
             "percent-positive.", "string"),
            ("ProvenanceType", "engineered", "string"),
            ("PredictedDirection", "negative", "string")],
      obj=[("mapsToConstruct", "HandoffsInformationExchange")])

out = "cckg-schema.owl"
# ------------------------------------------- v4: ConstructMeasurement (reified)
# Per-hospital-per-wave SOPS values are a property of the (SurveyEvent,
# SurveyConstruct) pair, not of either alone. Same reification discipline
# already used for ProxyMapping (bridge edges) and SurveyEvent (waves).
g.add((CCKG.ConstructMeasurement, RDF.type, OWL.Class))
g.add((CCKG.ConstructMeasurement, RDFS.comment, Literal(
    "One SOPS composite or single-item value for one hospital in one survey "
    "administration. Reified because the value belongs to the (SurveyEvent, "
    "SurveyConstruct) pair. Replaces the deprecated global PercentPositive "
    "slot on SurveyConstruct.")))

for _n, _d, _r, _c in [
    ("hasConstructMeasurement", "SurveyEvent", "ConstructMeasurement",
     "Links a survey administration to the values it reported."),
    ("measuresConstruct", "ConstructMeasurement", "SurveyConstruct",
     "Links a reported value to the construct it measures."),
]:
    _p = CCKG[_n]
    g.add((_p, RDF.type, OWL.ObjectProperty))
    g.add((_p, RDFS.domain, CCKG[_d]))
    g.add((_p, RDFS.range, CCKG[_r]))
    g.add((_p, RDFS.comment, Literal(_c)))

for _n, _t, _c in [
    ("MeasuredPercentPositive", "decimal",
     "Percent-positive response for this hospital-wave-construct, normalised "
     "to 0-100 across instruments."),
    ("ValueScale", "string", "Native published scale: 'percent' or 'proportion'."),
    ("SourceColumn", "string", "Source file column name, for provenance."),
    ("SourceFile", "string", "Source file name, for provenance."),
]:
    _p = CCKG[_n]
    g.add((_p, RDF.type, OWL.DatatypeProperty))
    g.add((_p, RDFS.domain, CCKG.ConstructMeasurement))
    g.add((_p, RDFS.range, XSD[_t]))
    g.add((_p, RDFS.comment, Literal(_c)))

# ------------------------------------------- v4: audit status on the mapping
# The Assumption / PredictedDirection of PM_UniqueStays_Handoffs are NOT
# edited: rewriting a pre-registered claim after population would void the
# pre-registration that gives the register its evidential force. The audit
# outcome is recorded as a separate annotation instead.
g.add((CCKG.PM_UniqueStays_Handoffs, RDFS.comment, Literal(
    "AUDIT STATUS: untestable-by-construction. UniqueStayCount is not "
    "computable (cross-patient simultaneity is foreclosed by per-patient date "
    "shifting) and is degenerate at its declared domain (one stay per "
    "EventAggregate). Reported in the construct-alignment register as "
    "untestable with reason; not silently dropped, not redefined.")))

# ------------------------------------------- v5: risk-score provenance edge
# Every other derived quantity in this schema carries explicit provenance;
# WellbeingRiskAssessment was the exception. assessedIn points at an
# ActivityWindow (per-stay, NOT per-caregiver) and assessmentOf at a Caregiver,
# so the EventAggregate a score was computed from was recoverable only by
# joining those two. That is an inconsistency, not a design choice: it made the
# risk score the one derived value whose inputs had to be reconstructed rather
# than read. assessmentFrom asserts it directly.
g.add((CCKG.assessmentFrom, RDF.type, OWL.ObjectProperty))
g.add((CCKG.assessmentFrom, RDFS.domain, CCKG.WellbeingRiskAssessment))
g.add((CCKG.assessmentFrom, RDFS.range, CCKG.EventAggregate))
g.add((CCKG.assessmentFrom, RDFS.comment, Literal(
    "The EventAggregate whose feature values were used to compute this "
    "assessment. Makes the score's provenance a single asserted edge rather "
    "than a join over assessedIn + assessmentOf.")))

# ------------------------------------------- v4: optional ClinicalActivity prune
if not MATERIALIZE_CLINICAL_ACTIVITY:
    _ca = CCKG.ClinicalActivity
    for _p in list(g.subjects(RDFS.domain, _ca)) + list(g.subjects(RDFS.range, _ca)):
        for _t in list(g.triples((_p, None, None))):
            g.remove(_t)
    for _t in list(g.triples((_ca, None, None))):
        g.remove(_t)
    print("ClinicalActivity and its properties removed (flag is False).")

g.add((ONT_IRI, OWL.versionInfo, Literal(
    "v5. Adds assessmentFrom (WellbeingRiskAssessment -> EventAggregate) so risk-score provenance is asserted, not reconstructed. v4 corrections from the v3 structural audit: (1) belongsToWindow domain "
    "ClinicalActivity -> EventAggregate, resolving a domain violation against "
    "populated data; (2) UniqueStayCount documented as declared-not-populated "
    "with both structural reasons; (3) PercentPositive deprecated in favour of "
    "ConstructMeasurement; (4) ConstructMeasurement class added for "
    "per-hospital-per-wave SOPS values; (5) ClinicalActivity flagged as "
    "declared-not-populated. Corrections are recorded, not silently applied.")))

g.serialize(destination=out, format="pretty-xml")
print(f"Serialized {len(g)} triples to {out}")
