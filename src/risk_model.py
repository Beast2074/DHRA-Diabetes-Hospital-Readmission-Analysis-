import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold, RandomizedSearchCV,
                                     cross_val_score, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import OUTPUT_DIR, DATA_PROCESSED, RANDOM_STATE, TOP_PCT
from src.db import query_df, get_connection
from src.utils import save_metrics
from src import plot_style as ps

NUM_COLS = [
    "age_mid", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "charlson_index", "n_active_meds", "n_dose_changes", "total_prior_visits",
    "prior_stays_in_network", "med_change", "diabetes_med",
]
CAT_COLS = [
    "admission_type", "admission_source", "discharge_group", "payer_code", "specialty_group",
    "diag1_group", "diag2_group", "diag3_group", "max_glu_serum", "a1c_result",
    "insulin_status", "metformin_status",
]
# race and gender are left out on purpose, they are only used for the fairness check

PRETTY = {
    "number_inpatient": "Prior inpatient stays",
    "discharge_group": "Discharge destination",
    "number_emergency": "Prior ER visits",
    "total_prior_visits": "Total prior visits",
    "prior_stays_in_network": "Earlier stays in this network",
    "number_diagnoses": "Number of diagnoses",
    "diag1_group": "Primary diagnosis",
    "diag2_group": "Secondary diagnosis",
    "diag3_group": "Third diagnosis",
    "age_mid": "Age",
    "time_in_hospital": "Length of stay",
    "num_medications": "Medications given",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures",
    "number_outpatient": "Prior outpatient visits",
    "specialty_group": "Admitting specialty",
    "payer_code": "Payer",
    "admission_source": "Admission source",
    "admission_type": "Admission type",
    "charlson_index": "Charlson comorbidity index",
    "insulin_status": "Insulin dosage",
    "metformin_status": "Metformin dosage",
    "a1c_result": "HbA1c result",
    "max_glu_serum": "Glucose serum result",
    "n_active_meds": "Diabetes drugs given",
    "n_dose_changes": "Diabetes dose changes",
    "med_change": "Any medication change",
    "diabetes_med": "On diabetes medication",
}


def prepare_features(df):
    df = df.copy()
    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    # 70+ specialties is too many, keep the 12 biggest
    top_specs = df["medical_specialty"].value_counts().head(12).index
    df["specialty_group"] = df["medical_specialty"].where(df["medical_specialty"].isin(top_specs), "Other")
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    df["age_band"] = pd.cut(df["age_mid"], bins=[0, 50, 70, 101], labels=["Under 50", "50-69", "70+"],
                            right=False).astype(str)
    return df


def top_k_stats(y, p, pct, seed=RANDOM_STATE):
    # flag the top pct of stays by risk, random tie break (matters for the integer LACE score)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(p) * pct))
    order = np.lexsort((rng.random(len(p)), -p))
    flagged = order[:k]
    captured = y[flagged].sum()
    precision = captured / k
    return {"precision": precision, "recall": captured / y.sum(), "lift": precision / y.mean()}


def bootstrap_auc(y, p_main, p_base, n_boot=500, seed=RANDOM_STATE):
    # paired bootstrap -> ci for the model auc and for (model - baseline)
    rng = np.random.default_rng(seed)
    aucs, diffs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        a = roc_auc_score(y[idx], p_main[idx])
        b = roc_auc_score(y[idx], p_base[idx])
        aucs.append(a)
        diffs.append(a - b)
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(diffs, [2.5, 97.5])


def evaluate(name, y, p, cv_scores):
    top20 = top_k_stats(y, p, TOP_PCT)
    top10 = top_k_stats(y, p, 0.10)
    return {
        "model": name,
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "test_auc": round(roc_auc_score(y, p), 4),
        "test_pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "precision_top20": round(top20["precision"], 4),
        "recall_top20": round(top20["recall"], 4),
        "lift_top20": round(top20["lift"], 2),
        "precision_top10": round(top10["precision"], 4),
        "recall_top10": round(top10["recall"], 4),
        "lift_top10": round(top10["lift"], 2),
    }


def decile_table(y, p):
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = d.groupby("decile", observed=True).agg(stays=("y", "size"), readmissions=("y", "sum"),
                                               avg_predicted=("p", "mean"))
    t["readmit_rate_pct"] = (t["readmissions"] / t["stays"] * 100).round(2)
    t["avg_predicted_pct"] = (t["avg_predicted"] * 100).round(2)
    t["lift"] = (t["readmissions"] / t["stays"] / d["y"].mean()).round(2)
    t["cum_pct_of_readmissions"] = (t["readmissions"].cumsum() / d["y"].sum() * 100).round(2)
    return t.drop(columns="avg_predicted").reset_index()


def fairness_audit(test_df, p, threshold):
    rows = []
    flagged = p >= threshold
    for col in ["race", "gender", "age_band"]:
        for group, idx in test_df.groupby(col).groups.items():
            pos = test_df.index.get_indexer(idx)
            yg = test_df["readmit_30"].values[pos]
            if len(yg) < 300 or yg.sum() < 20:
                continue
            fg = flagged[pos]
            rows.append({
                "attribute": col,
                "group": group,
                "stays": len(yg),
                "actual_rate_pct": round(yg.mean() * 100, 2),
                "flagged_pct": round(fg.mean() * 100, 2),
                "recall_pct": round(fg[yg == 1].mean() * 100, 2),
                "precision_pct": round(yg[fg].mean() * 100, 2) if fg.sum() else np.nan,
                "auc": round(roc_auc_score(yg, p[pos]), 4),
            })
    return pd.DataFrame(rows)


# charts
def chart_roc(y, preds, aucs):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    colors = {"Gradient boosting": ps.BLUE, "Logistic regression": ps.AQUA, "LACE index (baseline)": ps.ORANGE}
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=f"{name}  AUC {aucs[name]:.3f}")
    ax.plot([0, 1], [0, 1], color=ps.AXIS, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (readmissions caught)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("ML beats the clinical LACE score on unseen patients", pad=26)
    ps.subtitle(ax, "ROC curves on the 20% held-out test set (patients never seen in training)")
    return ps.save(fig, "06_model_roc_curves.png")


def chart_gains(y, preds):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    colors = {"Gradient boosting": ps.BLUE, "LACE index (baseline)": ps.ORANGE}
    x_pct = np.arange(1, 101)
    for name in colors:
        p = preds[name]
        caught = [top_k_stats(y, p, k / 100)["recall"] * 100 for k in x_pct]
        ax.plot(x_pct, caught, color=colors[name], linewidth=2, label=name)
        if name == "Gradient boosting":
            at20 = caught[int(TOP_PCT * 100) - 1]
            ax.scatter([TOP_PCT * 100], [at20], color=colors[name], s=50, zorder=4,
                       edgecolor=ps.SURFACE, linewidth=2)
            ax.annotate(f"top {TOP_PCT:.0%} of stays -> {at20:.0f}% of readmissions",
                        xy=(TOP_PCT * 100, at20), xytext=(12, -4), textcoords="offset points",
                        fontsize=9, color=ps.INK)
    ax.plot([0, 100], [0, 100], color=ps.AXIS, linewidth=1, label="Random selection")
    ax.set_xlabel("% of discharges targeted (highest risk first)")
    ax.set_ylabel("% of all readmissions captured")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Cumulative gains: who to call first", pad=26)
    ps.subtitle(ax, "Held-out test set. The steeper the curve, the fewer calls needed per readmission caught.")
    return ps.save(fig, "07_cumulative_gains.png")


def chart_deciles(dec, base_rate):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = dec["decile"].astype(int)
    colors = [ps.BLUE if d <= 2 else ps.LIGHT_GRAY for d in x]
    ax.bar(x, dec["readmit_rate_pct"], width=0.62, color=colors, zorder=2)
    ax.axhline(base_rate * 100, color=ps.AXIS, linewidth=1, zorder=1)
    for xi, r, lift in zip(x[:2], dec["readmit_rate_pct"][:2], dec["lift"][:2]):
        ax.text(xi, r + 0.6, f"{r:.1f}%\n{lift:.1f}x", ha="center", fontsize=8.5, color=ps.INK)
    ax.set_xticks(x)
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Actual 30-day readmission rate (%)")
    ax.set_ylim(0, dec["readmit_rate_pct"].max() + 6)
    ax.grid(axis="x", visible=False)
    ax.set_title("The top two risk deciles are where the readmissions are", pad=26)
    ps.subtitle(ax, f"Blue = the 20% the care team would call. Grey line = test set average {base_rate * 100:.1f}%.")
    return ps.save(fig, "08_risk_deciles.png")


def chart_calibration(y, p):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    lim = max(frac_pos.max(), mean_pred.max()) * 100 + 3
    ax.plot([0, lim], [0, lim], color=ps.AXIS, linewidth=1, label="Perfect calibration")
    ax.plot(mean_pred * 100, frac_pos * 100, color=ps.BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=ps.SURFACE, markeredgewidth=2, label="Gradient boosting")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted readmission risk (%)")
    ax.set_ylabel("Observed readmission rate (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Predicted risk matches reality", pad=26)
    ps.subtitle(ax, "Calibration by decile - needed because the cost model uses the probabilities")
    return ps.save(fig, "09_calibration.png")


def chart_importance(imp):
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    y = np.arange(len(top))
    ax.barh(y, top["auc_drop"], xerr=top["auc_drop_std"], height=0.6, color=ps.BLUE,
            error_kw={"ecolor": ps.INK_2, "elinewidth": 1, "capsize": 2}, zorder=2)
    ax.set_yticks(y, [PRETTY.get(f, f) for f in top["feature"]])
    ax.set_xlabel("Drop in test AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("Utilisation history and discharge plan drive the risk score", pad=26)
    ps.subtitle(ax, "Permutation importance on the held-out test set (5 repeats)")
    return ps.save(fig, "10_feature_importance.png")


def chart_fairness(fair, overall_auc):
    f = fair.copy()
    f["group"] = f["group"].replace({"AfricanAmerican": "African American"})
    f["label"] = (f["attribute"].map({"race": "Race", "gender": "Gender", "age_band": "Age"}) + ": "
                  + f["group"].astype(str) + "  (n=" + f["stays"].map("{:,}".format) + ")")
    f = f.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    y = np.arange(len(f))
    ax.axvline(overall_auc, color=ps.AXIS, linewidth=1, zorder=1)
    ax.scatter(f["auc"], y, color=ps.BLUE, s=55, zorder=3, edgecolor=ps.SURFACE, linewidth=2)
    for yi, a, r in zip(y, f["auc"], f["recall_pct"]):
        ax.text(a + 0.004, yi, f"AUC {a:.3f} | catches {r:.0f}%", va="center", fontsize=8.5, color=ps.INK_2)
    ax.set_yticks(y, f["label"])
    ax.set_xlim(f["auc"].min() - 0.03, f["auc"].max() + 0.06)
    ax.set_xlabel("Test AUC within the group")
    ax.grid(axis="y", visible=False)
    ax.set_title("Fairness check: steady across race and gender, weaker for 70+", pad=26)
    ps.subtitle(ax, f"Grey line = overall AUC {overall_auc:.3f}. 'Catches' = share of the group's readmissions flagged. Small n = noisy.")
    return ps.save(fig, "11_fairness_audit.png")


def write_scores_to_mysql(df, oof):
    # store out-of-fold risk so the care team can query it with sql
    pct = pd.Series(oof).rank(pct=True).values
    tier = np.where(pct > 0.90, "High", np.where(pct > 1 - TOP_PCT, "Elevated", "Standard"))
    percentile = np.minimum((pct * 100).astype(int) + 1, 100)
    rows = list(zip(df["encounter_id"].astype(int).tolist(), np.round(oof, 4).tolist(),
                    percentile.tolist(), tier.tolist(), ["hgb_v1"] * len(df)))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM risk_scores")
    sql = ("INSERT INTO risk_scores (encounter_id, risk_score, risk_percentile, risk_tier, model_version) "
           "VALUES (%s, %s, %s, %s, %s)")
    for i in range(0, len(rows), 5000):
        cur.executemany(sql, rows[i:i + 5000])
    conn.commit()
    cur.close()
    conn.close()
    return len(rows)


def main():
    start = time.time()
    ps.set_style()
    df = prepare_features(query_df("SELECT * FROM vw_model_dataset"))
    X = df[NUM_COLS + CAT_COLS]
    y = df["readmit_30"].values
    groups = df["patient_nbr"].values

    # split by patient so the same person is never in both train and test (avoids leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    assert not set(groups[train_idx]) & set(groups[test_idx])
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]
    print(f"train {len(train_idx):,} stays / test {len(test_idx):,} stays | test base rate {y_test.mean():.2%}")

    cv = GroupKFold(n_splits=5)

    # 1) baseline: LACE score turned into a probability with a 1 variable logistic model
    lace_train = df.iloc[train_idx][["lace_score"]]
    lace_test = df.iloc[test_idx][["lace_score"]]
    lace_model = LogisticRegression()
    lace_cv = cross_val_score(lace_model, lace_train, y_train, groups=g_train, cv=cv, scoring="roc_auc")
    lace_model.fit(lace_train, y_train)
    p_lace = lace_model.predict_proba(lace_test)[:, 1]

    # 2) logistic regression with one hot encoding
    lr = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CAT_COLS),
        ])),
        ("model", LogisticRegression(max_iter=3000, C=0.3)),
    ])
    lr_cv = cross_val_score(lr, X_train, y_train, groups=g_train, cv=cv, scoring="roc_auc")
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    print(f"logistic regression cv auc {lr_cv.mean():.4f}")

    # 3) gradient boosting, tuned with grouped cross validation
    hgb = HistGradientBoostingClassifier(categorical_features="from_dtype", early_stopping=False,
                                         random_state=RANDOM_STATE)
    params = {
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_iter": [300, 500, 800],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [100, 200, 400, 600, 800],
        "l2_regularization": [0.0, 1.0, 5.0, 10.0],
        "max_features": [0.5, 0.8, 1.0],
    }
    search = RandomizedSearchCV(hgb, params, n_iter=20, scoring="roc_auc", cv=cv, n_jobs=-1,
                                random_state=RANDOM_STATE)
    search.fit(X_train, y_train, groups=g_train)
    best = search.best_estimator_
    cv_scores = np.array([search.cv_results_[f"split{i}_test_score"][search.best_index_] for i in range(5)])
    p_hgb = best.predict_proba(X_test)[:, 1]
    print(f"gradient boosting best cv auc {cv_scores.mean():.4f} with {search.best_params_}")

    # evaluation
    preds = {"Gradient boosting": p_hgb, "Logistic regression": p_lr, "LACE index (baseline)": p_lace}
    metrics = pd.DataFrame([
        evaluate("Gradient boosting", y_test, p_hgb, cv_scores),
        evaluate("Logistic regression", y_test, p_lr, lr_cv),
        evaluate("LACE index (baseline)", y_test, p_lace, lace_cv),
    ])
    (auc_lo, auc_hi), (diff_lo, diff_hi) = bootstrap_auc(y_test, p_hgb, p_lace)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    print(metrics[["model", "cv_auc_mean", "test_auc", "test_pr_auc", "recall_top20", "lift_top10"]].to_string(index=False))
    print(f"gb auc 95% ci [{auc_lo:.3f}, {auc_hi:.3f}] | gain over LACE 95% ci [{diff_lo:.3f}, {diff_hi:.3f}]")

    # classic LACE >= 10 rule for comparison
    lace_rule = lace_test["lace_score"].values >= 10
    lace_rule_stats = {
        "flagged_pct": round(lace_rule.mean() * 100, 2),
        "precision_pct": round(y_test[lace_rule].mean() * 100, 2),
        "recall_pct": round(y_test[lace_rule].sum() / y_test.sum() * 100, 2),
    }

    dec = decile_table(y_test, p_hgb)
    dec.to_csv(OUTPUT_DIR / "model_risk_deciles.csv", index=False)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({"feature": X_test.columns, "auc_drop": imp.importances_mean,
                           "auc_drop_std": imp.importances_std}).sort_values("auc_drop", ascending=False)
    imp_df.round(5).to_csv(OUTPUT_DIR / "model_feature_importance.csv", index=False)

    test_df = df.iloc[test_idx].reset_index(drop=True)
    threshold = np.quantile(p_hgb, 1 - TOP_PCT)
    fair = fairness_audit(test_df, p_hgb, threshold)
    fair.to_csv(OUTPUT_DIR / "model_fairness_audit.csv", index=False)

    # test predictions are used by the cost-benefit step
    pd.DataFrame({
        "encounter_id": test_df["encounter_id"],
        "patient_nbr": test_df["patient_nbr"],
        "readmit_30": y_test,
        "p_hgb": p_hgb,
        "p_lr": p_lr,
        "p_lace": p_lace,
        "lace_score": test_df["lace_score"],
    }).to_csv(DATA_PROCESSED / "test_predictions.csv", index=False)

    # charts
    aucs = dict(zip(metrics["model"], metrics["test_auc"]))
    chart_roc(y_test, preds, aucs)
    chart_gains(y_test, preds)
    chart_deciles(dec, y_test.mean())
    chart_calibration(y_test, p_hgb)
    chart_importance(imp_df)
    chart_fairness(fair, aucs["Gradient boosting"])

    # score every stay out-of-fold and push to mysql
    oof = cross_val_predict(clone(best), X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    n_scored = write_scores_to_mysql(df, oof)
    print(f"wrote {n_scored:,} out-of-fold risk scores to mysql (oof auc {roc_auc_score(y, oof):.4f})")

    hgb_row = metrics.iloc[0]
    lace_row = metrics.iloc[2]
    save_metrics("model", {
        "train_stays": int(len(train_idx)),
        "test_stays": int(len(test_idx)),
        "test_base_rate_pct": round(y_test.mean() * 100, 2),
        "best_params": search.best_params_,
        "hgb_cv_auc": float(hgb_row["cv_auc_mean"]),
        "hgb_test_auc": float(hgb_row["test_auc"]),
        "hgb_auc_ci": [round(auc_lo, 4), round(auc_hi, 4)],
        "lr_test_auc": float(metrics.iloc[1]["test_auc"]),
        "lace_test_auc": float(lace_row["test_auc"]),
        "auc_gain_vs_lace_ci": [round(diff_lo, 4), round(diff_hi, 4)],
        "hgb_recall_top20_pct": round(hgb_row["recall_top20"] * 100, 2),
        "lace_recall_top20_pct": round(lace_row["recall_top20"] * 100, 2),
        "hgb_precision_top20_pct": round(hgb_row["precision_top20"] * 100, 2),
        "hgb_lift_top10": float(hgb_row["lift_top10"]),
        "hgb_top_decile_rate_pct": float(dec.iloc[0]["readmit_rate_pct"]),
        "lace_rule_ge10": lace_rule_stats,
        "oof_auc": round(roc_auc_score(y, oof), 4),
        "stays_scored_in_mysql": n_scored,
        "top_features": imp_df["feature"].head(5).tolist(),
        "fairness_auc_range": [float(fair["auc"].min()), float(fair["auc"].max())],
    })
    print(f"model step done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
