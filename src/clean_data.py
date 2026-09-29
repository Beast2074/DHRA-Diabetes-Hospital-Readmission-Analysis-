import csv

import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, OUTPUT_DIR, EXCLUDED_DISCHARGE_IDS
from src.clinical_scores import icd9_group, charlson_conditions, charlson_index, lace_score

MED_CLASSES = {
    "metformin": "Biguanide",
    "repaglinide": "Meglitinide",
    "nateglinide": "Meglitinide",
    "chlorpropamide": "Sulfonylurea",
    "glimepiride": "Sulfonylurea",
    "acetohexamide": "Sulfonylurea",
    "glipizide": "Sulfonylurea",
    "glyburide": "Sulfonylurea",
    "tolbutamide": "Sulfonylurea",
    "tolazamide": "Sulfonylurea",
    "pioglitazone": "Thiazolidinedione",
    "rosiglitazone": "Thiazolidinedione",
    "troglitazone": "Thiazolidinedione",
    "acarbose": "Alpha-glucosidase inhibitor",
    "miglitol": "Alpha-glucosidase inhibitor",
    "examide": "Other",
    "citoglipton": "Other",
    "insulin": "Insulin",
    "glyburide-metformin": "Combination",
    "glipizide-metformin": "Combination",
    "glimepiride-pioglitazone": "Combination",
    "metformin-rosiglitazone": "Combination",
    "metformin-pioglitazone": "Combination",
}
MED_COLS = list(MED_CLASSES.keys())

# my grouping of the id codes into something readable
ADMISSION_TYPE_GROUP = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Other", 7: "Emergency"}
ADMISSION_SOURCE_GROUP = {
    7: "Emergency Room", 1: "Referral", 2: "Referral", 3: "Referral",
    4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer", 18: "Transfer",
    22: "Transfer", 25: "Transfer", 26: "Transfer",
    9: "Unknown", 15: "Unknown", 17: "Unknown", 20: "Unknown", 21: "Unknown",
}
DISCHARGE_GROUP = {
    1: "Home", 6: "Home Health", 8: "Home Health",
    3: "SNF / Nursing", 4: "SNF / Nursing", 15: "SNF / Nursing", 24: "SNF / Nursing",
    22: "Rehab / LTC", 23: "Rehab / LTC",
    2: "Other Hospital", 29: "Other Hospital",
    5: "Other Facility", 27: "Other Facility", 28: "Other Facility", 30: "Other Facility",
    7: "Left AMA",
    11: "Expired / Hospice", 13: "Expired / Hospice", 14: "Expired / Hospice",
    19: "Expired / Hospice", 20: "Expired / Hospice", 21: "Expired / Hospice",
}


def read_id_mapping():
    # IDS_mapping.csv is actually 3 tables stacked with blank rows in between
    tables = []
    current = []
    with open(DATA_RAW / "IDS_mapping.csv", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or all(cell.strip() == "" for cell in row):
                if current:
                    tables.append(current)
                current = []
            else:
                current.append(row)
    if current:
        tables.append(current)

    out = []
    for t in tables:
        df = pd.DataFrame(t[1:], columns=["id", "description"])
        df["id"] = df["id"].astype(int)
        df["description"] = df["description"].str.strip().replace({"NULL": "Unknown (NULL)"})
        out.append(df)
    return out  # admission_type, discharge_disposition, admission_source


def data_quality_report(df):
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(df[col].isna().mean() * 100, 2),
            "unique_values": int(df[col].nunique()),
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # "?" is how this dataset marks missing values
    df = pd.read_csv(
        DATA_RAW / "diabetic_data.csv",
        na_values=["?"],
        keep_default_na=False,
        low_memory=False,
        dtype={"diag_1": "string", "diag_2": "string", "diag_3": "string"},
    )
    print("raw shape:", df.shape)

    # data quality checks
    dq = data_quality_report(df)
    dq.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

    checks = []
    checks.append(("raw encounters", len(df)))
    checks.append(("unique patients", df["patient_nbr"].nunique()))
    checks.append(("duplicate encounter_id", int(df["encounter_id"].duplicated().sum())))
    checks.append(("invalid gender rows (dropped)", int((df["gender"] == "Unknown/Invalid").sum())))
    checks.append(("weight missing % (column dropped)", round(df["weight"].isna().mean() * 100, 1)))
    checks.append(("payer_code missing % (set to Unknown)", round(df["payer_code"].isna().mean() * 100, 1)))
    checks.append(("medical_specialty missing % (set to Unknown)", round(df["medical_specialty"].isna().mean() * 100, 1)))
    checks.append(("race missing % (set to Unknown)", round(df["race"].isna().mean() * 100, 1)))
    checks.append(("expired / hospice discharges (kept in db, excluded from analysis)",
                   int(df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).sum())))
    checks.append(("patients with more than one encounter",
                   int((df["patient_nbr"].value_counts() > 1).sum())))

    # cleaning
    df = df[df["gender"] != "Unknown/Invalid"].copy()
    df = df.drop(columns=["weight"])
    df["race"] = df["race"].fillna("Unknown")
    df["payer_code"] = df["payer_code"].fillna("Unknown")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df["max_glu_serum"] = df["max_glu_serum"].replace({"None": "Not Tested"})
    df["A1Cresult"] = df["A1Cresult"].replace({"None": "Not Tested"})

    # age comes as "[70-80)" -> "70-80" and a numeric midpoint
    df["age_group"] = df["age"].str.strip("[)")
    df["age_mid"] = df["age_group"].str.split("-").str[0].astype(int) + 5

    df["med_change"] = (df["change"] == "Ch").astype(int)
    df["diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)

    # clinical scores
    all_codes = pd.unique(df[["diag_1", "diag_2", "diag_3"]].values.ravel())
    all_codes = [c for c in all_codes if not pd.isna(c)]
    charlson_lookup = {c: charlson_conditions(c) for c in all_codes}

    diag_values = df[["diag_1", "diag_2", "diag_3"]].to_numpy(dtype=object)
    df["charlson_index"] = [charlson_index(row, charlson_lookup) for row in diag_values]
    df["lace_score"] = [
        lace_score(los, at, src, cci, ed)
        for los, at, src, cci, ed in zip(
            df["time_in_hospital"], df["admission_type_id"], df["admission_source_id"],
            df["charlson_index"], df["number_emergency"],
        )
    ]

    # lookup tables
    adm_type, discharge, adm_source = read_id_mapping()

    adm_type = adm_type.rename(columns={"id": "admission_type_id"})
    adm_type["type_group"] = adm_type["admission_type_id"].map(ADMISSION_TYPE_GROUP).fillna("Unknown")

    adm_source = adm_source.rename(columns={"id": "admission_source_id"})
    adm_source["source_group"] = adm_source["admission_source_id"].map(ADMISSION_SOURCE_GROUP).fillna("Other")

    discharge = discharge.rename(columns={"id": "discharge_disposition_id"})
    discharge["disposition_group"] = discharge["discharge_disposition_id"].map(DISCHARGE_GROUP).fillna("Other / Unknown")
    discharge["is_expired_or_hospice"] = discharge["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).astype(int)

    meds = pd.DataFrame({
        "medication_id": range(1, len(MED_COLS) + 1),
        "medication_name": MED_COLS,
        "drug_class": [MED_CLASSES[m] for m in MED_COLS],
    })

    # patients
    # race / gender should not change between visits, check it and keep the first visit value
    df = df.sort_values("encounter_id")
    inconsistent = df.groupby("patient_nbr")[["race", "gender"]].nunique()
    checks.append(("patients with conflicting race/gender across visits",
                   int((inconsistent.max(axis=1) > 1).sum())))
    patients = df.groupby("patient_nbr", as_index=False)[["race", "gender"]].first()

    # encounters
    enc_cols = [
        "encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "age_group", "age_mid", "time_in_hospital", "payer_code",
        "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
        "max_glu_serum", "A1Cresult", "med_change", "diabetes_med", "charlson_index",
        "lace_score", "readmitted", "readmit_30",
    ]
    encounters = df[enc_cols].rename(columns={"A1Cresult": "a1c_result"})

    # diagnoses (wide -> long)
    diag = df.melt(id_vars="encounter_id", value_vars=["diag_1", "diag_2", "diag_3"],
                   var_name="diag_position", value_name="icd9_code")
    diag = diag.dropna(subset=["icd9_code"])
    diag["diag_position"] = diag["diag_position"].str[-1].astype(int)
    group_lookup = {c: icd9_group(c) for c in all_codes}
    diag["diag_group"] = diag["icd9_code"].map(group_lookup)
    diag = diag.sort_values(["encounter_id", "diag_position"])

    # medications (wide -> long, keep only drugs actually given)
    med_long = df.melt(id_vars="encounter_id", value_vars=MED_COLS,
                       var_name="medication_name", value_name="dosage_status")
    med_long = med_long[med_long["dosage_status"] != "No"]
    med_long = med_long.merge(meds[["medication_id", "medication_name"]], on="medication_name")
    med_long = med_long[["encounter_id", "medication_id", "dosage_status"]].sort_values(["encounter_id", "medication_id"])

    checks.append(("clean encounters", len(encounters)))
    checks.append(("encounters eligible for analysis (not expired/hospice)",
                   int((~encounters["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)).sum())))
    checks.append(("30-day readmission rate % (eligible)",
                   round(encounters.loc[~encounters["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS), "readmit_30"].mean() * 100, 2)))
    checks.append(("HbA1c tested % of encounters", round((encounters["a1c_result"] != "Not Tested").mean() * 100, 2)))

    # save
    adm_type.to_csv(DATA_PROCESSED / "admission_type.csv", index=False)
    adm_source.to_csv(DATA_PROCESSED / "admission_source.csv", index=False)
    discharge.to_csv(DATA_PROCESSED / "discharge_disposition.csv", index=False)
    meds.to_csv(DATA_PROCESSED / "medications.csv", index=False)
    patients.to_csv(DATA_PROCESSED / "patients.csv", index=False)
    encounters.to_csv(DATA_PROCESSED / "encounters.csv", index=False)
    diag.to_csv(DATA_PROCESSED / "encounter_diagnoses.csv", index=False)
    med_long.to_csv(DATA_PROCESSED / "encounter_medications.csv", index=False)

    checks_df = pd.DataFrame(checks, columns=["check", "result"], dtype=object)
    checks_df.to_csv(OUTPUT_DIR / "data_quality_checks.csv", index=False)

    print(checks_df.to_string(index=False))
    print(f"tables -> patients {len(patients):,} | encounters {len(encounters):,} | "
          f"diagnoses {len(diag):,} | medications {len(med_long):,}")


if __name__ == "__main__":
    main()
