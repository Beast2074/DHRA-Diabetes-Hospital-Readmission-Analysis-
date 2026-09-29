import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from config import OUTPUT_DIR, SQL_RESULTS_DIR
from src.db import query_df
from src import plot_style as ps


def add_helper_cols(df):
    df = df.copy()
    df["prior_inpatient"] = df["number_inpatient"].clip(upper=5)
    df["prior_emergency"] = df["number_emergency"].clip(upper=4)
    df["a1c_tested"] = (df["a1c_result"] != "Not Tested").astype(int)
    df["meds_per_10"] = df["num_medications"] / 10
    df["age_band"] = pd.cut(df["age_mid"], bins=[0, 50, 60, 70, 80, 101],
                            labels=["<50", "50-59", "60-69", "70-79", "80+"], right=False).astype(str)
    return df


def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def run_chi_square(df, factors):
    rows = []
    alpha = 0.05 / len(factors)  # bonferroni because we run many tests
    for f in factors:
        table = pd.crosstab(df[f], df["readmit_30"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rate = df.groupby(f)["readmit_30"].mean()
        rows.append({
            "factor": f,
            "chi2": round(chi2, 1),
            "dof": dof,
            "p_value": p,
            "cramers_v": round(cramers_v(table), 4),
            "significant_bonferroni": p < alpha,
            "highest_rate_group": rate.idxmax(),
            "highest_rate_pct": round(rate.max() * 100, 2),
            "lowest_rate_group": rate.idxmin(),
            "lowest_rate_pct": round(rate.min() * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)


def run_mann_whitney(df, cols):
    rows = []
    yes = df[df["readmit_30"] == 1]
    no = df[df["readmit_30"] == 0]
    for c in cols:
        u, p = stats.mannwhitneyu(yes[c], no[c], alternative="two-sided")
        # rank biserial correlation as effect size
        effect = 1 - 2 * u / (len(yes) * len(no))
        rows.append({
            "variable": c,
            "mean_readmitted": round(yes[c].mean(), 2),
            "mean_not_readmitted": round(no[c].mean(), 2),
            "p_value": p,
            "rank_biserial": round(-effect, 4),
        })
    return pd.DataFrame(rows)


def run_logit(df):
    formula = (
        "readmit_30 ~ prior_inpatient + prior_emergency + number_outpatient + time_in_hospital"
        " + meds_per_10 + number_diagnoses + a1c_tested"
        " + C(age_band, Treatment('50-59'))"
        " + C(discharge_group, Treatment('Home'))"
        " + C(admission_source, Treatment('Referral'))"
        " + C(diag1_group, Treatment('Circulatory'))"
        " + C(insulin_status, Treatment('No'))"
        " + diabetes_med"
    )
    model = smf.logit(formula, data=df).fit(disp=0)
    ci = model.conf_int()
    out = pd.DataFrame({
        "term": model.params.index,
        "odds_ratio": np.exp(model.params.values),
        "ci_low": np.exp(ci[0].values),
        "ci_high": np.exp(ci[1].values),
        "p_value": model.pvalues.values,
    })
    out = out[out["term"] != "Intercept"]
    # make the patsy names readable
    out["term"] = (out["term"]
                   .str.replace(r"C\((\w+), Treatment\('[^']+'\)\)\[T\.", r"\1 = ", regex=True)
                   .str.replace("]", "", regex=False))
    return out, model


def chart_prior_admissions(df, base_rate):
    g = df.groupby("prior_inpatient")["readmit_30"].agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
    labels = ["0", "1", "2", "3", "4", "5+"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(g))
    ax.bar(x, g["rate"] * 100, width=0.6, color=ps.BLUE, zorder=2)
    ax.errorbar(x, g["rate"] * 100, yerr=[(g["rate"] - lo) * 100, (hi - g["rate"]) * 100],
                fmt="none", ecolor=ps.INK_2, elinewidth=1, capsize=3, zorder=3)
    ax.axhline(base_rate * 100, color=ps.AXIS, linewidth=1, zorder=1)
    for i, r in enumerate(g["rate"]):
        ax.text(i, hi.iloc[i] * 100 + 0.8, f"{r * 100:.1f}%", ha="center", fontsize=9, color=ps.INK)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Inpatient stays in the previous 12 months")
    ax.set_ylabel("30-day readmission rate (%)")
    ax.set_ylim(0, (hi.max() * 100) + 6)
    ax.grid(axis="x", visible=False)
    ax.set_title("Readmission risk climbs with every prior hospital stay", pad=26)
    ps.subtitle(ax, f"5+ prior stays = 4.3x the risk of none. Grey line = network average {base_rate * 100:.1f}%, whiskers = 95% Wilson CI.")
    ps.source_note(fig)
    return ps.save(fig, "01_prior_admissions_dose_response.png")


def chart_discharge(df, base_rate):
    g = df.groupby("discharge_group")["readmit_30"].agg(["mean", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [ps.BLUE if m > base_rate else ps.LIGHT_GRAY for m in g["mean"]]
    y = np.arange(len(g))
    ax.barh(y, g["mean"] * 100, height=0.6, color=colors, zorder=2)
    ax.axvline(base_rate * 100, color=ps.AXIS, linewidth=1, zorder=1)
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(m * 100 + 0.3, i, f"{m * 100:.1f}%  (n={n:,})", va="center", fontsize=8.5, color=ps.INK_2)
    ax.set_yticks(y, g.index)
    ax.set_xlabel("30-day readmission rate (%)")
    ax.set_xlim(0, g["mean"].max() * 100 + 8)
    ax.grid(axis="y", visible=False)
    ax.set_title("Patients sent to rehab or another facility come back most", pad=26)
    ps.subtitle(ax, f"Blue = above the {base_rate * 100:.1f}% network average (grey line). Rehab / LTC is 2.6x the home rate.")
    ps.source_note(fig)
    return ps.save(fig, "02_discharge_destination.png")


def chart_hba1c_gap():
    q = pd.read_csv(SQL_RESULTS_DIR / "q06_hba1c_care_gap.csv")
    q = q.sort_values("readmit_rate_not_tested")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(q))
    ax.hlines(y, q["readmit_rate_tested"], q["readmit_rate_not_tested"], color=ps.AXIS, linewidth=2, zorder=1)
    ax.scatter(q["readmit_rate_tested"], y, s=60, color=ps.BLUE, zorder=3, label="HbA1c tested",
               edgecolor=ps.SURFACE, linewidth=2)
    ax.scatter(q["readmit_rate_not_tested"], y, s=60, color=ps.ORANGE, zorder=3, label="Not tested",
               edgecolor=ps.SURFACE, linewidth=2)
    labels = [f"{d}  ({t:.0f}% tested)" for d, t in zip(q["primary_diagnosis"], q["hba1c_tested_pct"])]
    ax.set_yticks(y, labels)
    ax.set_xlabel("30-day readmission rate (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    diab = q[q["primary_diagnosis"] == "Diabetes"].iloc[0]
    ax.annotate(f"{diab['readmit_rate_not_tested']:.1f}% vs {diab['readmit_rate_tested']:.1f}%",
                xy=(diab["readmit_rate_not_tested"], y[list(q["primary_diagnosis"]).index("Diabetes")]),
                xytext=(8, -3), textcoords="offset points", fontsize=8.5, color=ps.INK)
    ax.set_title("Only 1 in 6 stays gets an HbA1c test; untested diabetes patients return 53% more", pad=26)
    ps.subtitle(ax, "Readmission rate by primary diagnosis, tested vs not tested for HbA1c during the stay")
    ps.source_note(fig)
    return ps.save(fig, "03_hba1c_care_gap.png")


def chart_repeat_patients():
    q = pd.read_csv(SQL_RESULTS_DIR / "q10_frequent_flyers.csv")
    shades = {"1 stay": ps.BLUE_RAMP[2], "2 stays": ps.BLUE_RAMP[4], "3+ stays": ps.BLUE_RAMP[6]}
    rows = [("Share of patients", "pct_of_patients"),
            ("Share of encounters", "pct_of_encounters"),
            ("Share of readmissions", "pct_of_readmissions")]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, col) in enumerate(rows):
        left = 0
        for _, r in q.iterrows():
            w = r[col]
            ax.barh(i, w - 0.3, left=left + 0.15, height=0.55, color=shades[r["patient_type"]], zorder=2,
                    label=r["patient_type"] if i == 0 else None)
            if w > 7:
                txt_color = "white" if r["patient_type"] != "1 stay" else ps.INK
                ax.text(left + w / 2, i, f"{w:.1f}%", ha="center", va="center", fontsize=9, color=txt_color)
            left += w
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.grid(False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=9)
    ax.set_title("9% of patients generate over half of all readmissions", pad=26)
    ps.subtitle(ax, "Patients split by how many eligible stays they had in the data")
    return ps.save(fig, "04_repeat_patients_concentration.png")


PRETTY_TERMS = {
    "prior_inpatient": "Each prior inpatient stay",
    "prior_emergency": "Each prior ER visit",
    "number_outpatient": "Each prior outpatient visit",
    "time_in_hospital": "Each extra day in hospital",
    "meds_per_10": "Every 10 extra medications",
    "number_diagnoses": "Each extra diagnosis coded",
    "a1c_tested": "HbA1c tested during stay",
    "diabetes_med": "On any diabetes medication",
}
PRETTY_PREFIX = {
    "discharge_group": ("Discharged to", "vs home"),
    "age_band": ("Age", "vs 50-59"),
    "admission_source": ("Admitted via", "vs referral"),
    "diag1_group": ("Primary dx", "vs circulatory"),
    "insulin_status": ("Insulin dose", "vs no insulin"),
}


def pretty_term(term):
    if term in PRETTY_TERMS:
        return PRETTY_TERMS[term]
    col, _, level = term.partition(" = ")
    if col in PRETTY_PREFIX:
        prefix, ref = PRETTY_PREFIX[col]
        return f"{prefix} {level} ({ref})"
    return term


def chart_forest(or_table):
    keep = or_table[or_table["p_value"] < 0.05].copy()
    keep = keep.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(keep) + 1.5)))
    y = np.arange(len(keep))
    colors = [ps.ORANGE if o > 1 else ps.BLUE for o in keep["odds_ratio"]]
    ax.hlines(y, keep["ci_low"], keep["ci_high"], color=ps.INK_2, linewidth=1.2, zorder=2)
    ax.scatter(keep["odds_ratio"], y, color=colors, s=40, zorder=3, edgecolor=ps.SURFACE, linewidth=1.5)
    ax.axvline(1, color=ps.AXIS, linewidth=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3, 4], ["0.5", "0.75", "1", "1.5", "2", "3", "4"])
    ax.minorticks_off()
    ax.set_yticks(y, [pretty_term(t) for t in keep["term"]])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("Adjusted odds ratio (log scale, 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What still matters after adjusting for everything else", pad=26)
    ps.subtitle(ax, "Multivariable logistic regression on first stays, only p < 0.05 shown. Orange = raises risk, blue = lowers it.")
    return ps.save(fig, "05_adjusted_odds_ratios.png")


def main():
    ps.set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # all eligible stays for descriptive charts, first stay per patient for the stats tests
    df = add_helper_cols(query_df("SELECT * FROM vw_model_dataset"))
    first_ids = query_df("SELECT encounter_id FROM vw_first_encounters")["encounter_id"]
    first = df[df["encounter_id"].isin(first_ids)]
    base_rate = df["readmit_30"].mean()
    print(f"eligible stays: {len(df):,} | first stays: {len(first):,} | base rate {base_rate:.2%}")

    # hypothesis tests
    factors = ["prior_inpatient", "discharge_group", "age_band", "admission_source", "admission_type",
               "diag1_group", "a1c_result", "insulin_status", "med_change", "diabetes_med", "race", "gender"]
    chi = run_chi_square(first, factors)
    chi.to_csv(OUTPUT_DIR / "stats_chi_square_tests.csv", index=False)
    print(chi[["factor", "p_value", "cramers_v", "significant_bonferroni"]].to_string(index=False))

    mw = run_mann_whitney(first, ["time_in_hospital", "num_medications", "number_diagnoses",
                                  "num_lab_procedures", "number_emergency", "number_inpatient"])
    mw.to_csv(OUTPUT_DIR / "stats_mann_whitney_tests.csv", index=False)

    # hba1c test vs readmission for patients admitted mainly for diabetes (two proportion z test)
    diab = first[first["diag1_group"] == "Diabetes"]
    counts = diab.groupby("a1c_tested")["readmit_30"].agg(["sum", "count"])
    z, p = proportions_ztest(counts["sum"].values, counts["count"].values)
    rates = counts["sum"] / counts["count"]
    hba1c_test = pd.DataFrame([{
        "group": "primary diagnosis = diabetes (first stays)",
        "n_not_tested": int(counts.loc[0, "count"]),
        "rate_not_tested_pct": round(rates[0] * 100, 2),
        "n_tested": int(counts.loc[1, "count"]),
        "rate_tested_pct": round(rates[1] * 100, 2),
        "z_stat": round(z, 3),
        "p_value": p,
    }])
    hba1c_test.to_csv(OUTPUT_DIR / "stats_hba1c_ztest.csv", index=False)
    print(hba1c_test.to_string(index=False))

    # multivariable logistic regression
    or_table, model = run_logit(first)
    or_table.round({"odds_ratio": 3, "ci_low": 3, "ci_high": 3}).to_csv(
        OUTPUT_DIR / "stats_adjusted_odds_ratios.csv", index=False)
    print(f"logit pseudo r2 = {model.prsquared:.4f}, n = {int(model.nobs):,}")

    # charts
    chart_prior_admissions(df, base_rate)
    chart_discharge(df, base_rate)
    chart_hba1c_gap()
    chart_repeat_patients()
    chart_forest(or_table)
    print("eda charts saved")


if __name__ == "__main__":
    main()
