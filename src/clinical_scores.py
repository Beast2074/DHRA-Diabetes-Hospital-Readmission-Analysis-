import pandas as pd

# icd-9 grouping follows Strack et al. 2014 (the paper that released this dataset)
# charlson + lace are standard clinical scores, rebuilt here from the 3 diagnosis codes we have

CHARLSON_WEIGHTS = {
    "mi": 1, "chf": 1, "pvd": 1, "cvd": 1, "dementia": 1, "copd": 1,
    "rheum": 1, "ulcer": 1, "mild_liver": 1, "diabetes": 1,
    "diabetes_comp": 2, "hemiplegia": 2, "renal": 2, "cancer": 2,
    "severe_liver": 3, "metastatic": 6, "aids": 6,
}


def icd9_group(code):
    if pd.isna(code):
        return None
    code = str(code)
    # V codes (supplementary) and E codes (external injury) go to other
    if code[0] in ("V", "E"):
        return "Other"
    num = float(code)
    whole = int(num)
    if whole == 250:
        return "Diabetes"
    if 390 <= whole <= 459 or whole == 785:
        return "Circulatory"
    if 460 <= whole <= 519 or whole == 786:
        return "Respiratory"
    if 520 <= whole <= 579 or whole == 787:
        return "Digestive"
    if 800 <= whole <= 999:
        return "Injury"
    if 710 <= whole <= 739:
        return "Musculoskeletal"
    if 580 <= whole <= 629 or whole == 788:
        return "Genitourinary"
    if 140 <= whole <= 239:
        return "Neoplasms"
    return "Other"


def charlson_conditions(code):
    # returns the charlson conditions a single icd-9 code maps to (deyo version, simplified)
    found = set()
    if pd.isna(code) or str(code)[0] in ("V", "E"):
        return found
    code = str(code)
    num = float(code)
    whole = int(num)

    if whole in (410, 412):
        found.add("mi")
    if whole == 428:
        found.add("chf")
    if whole == 441 or 443.9 <= num < 444 or 785.4 <= num < 785.5:
        found.add("pvd")
    if 430 <= whole <= 438:
        found.add("cvd")
    if whole == 290:
        found.add("dementia")
    if 490 <= whole <= 505 or 506.4 <= num < 506.5:
        found.add("copd")
    if whole in (710, 714, 725):
        found.add("rheum")
    if 531 <= whole <= 534:
        found.add("ulcer")
    if whole == 571:
        found.add("mild_liver")
    if whole == 572:
        found.add("severe_liver")
    if whole == 250:
        # 4th digit 4-6 = diabetes with organ damage
        fourth = code.split(".")[1][0] if "." in code else "0"
        if fourth in ("4", "5", "6"):
            found.add("diabetes_comp")
        else:
            found.add("diabetes")
    if whole in (342, 344):
        found.add("hemiplegia")
    if whole in (582, 583, 585, 586, 588):
        found.add("renal")
    if 140 <= whole <= 172 or 174 <= whole <= 195 or 200 <= whole <= 208:
        found.add("cancer")
    if 196 <= whole <= 199:
        found.add("metastatic")
    if 42 <= whole <= 44:
        found.add("aids")
    return found


def charlson_index(codes, lookup):
    # codes = list of up to 3 icd codes for one encounter, lookup = code -> set cache
    conditions = set()
    for c in codes:
        if not pd.isna(c):
            conditions |= lookup.get(c, set())
    # hierarchy rules so the same disease isn't counted twice
    if "diabetes_comp" in conditions:
        conditions.discard("diabetes")
    if "metastatic" in conditions:
        conditions.discard("cancer")
    if "severe_liver" in conditions:
        conditions.discard("mild_liver")
    return sum(CHARLSON_WEIGHTS[c] for c in conditions)


def lace_score(los, admission_type_id, admission_source_id, cci, ed_visits):
    # L = length of stay
    if los < 1:
        l_pts = 0
    elif los <= 3:
        l_pts = int(los)
    elif los <= 6:
        l_pts = 4
    elif los <= 13:
        l_pts = 5
    else:
        l_pts = 7
    # A = acute / emergency admission
    a_pts = 3 if admission_type_id in (1, 2, 7) or admission_source_id == 7 else 0
    # C = charlson, 4+ gets 5 points
    c_pts = cci if cci <= 3 else 5
    # E = prior ED visits (dataset has prior year, lace uses 6 months -> proxy)
    e_pts = min(int(ed_visits), 4)
    return l_pts + a_pts + c_pts + e_pts
