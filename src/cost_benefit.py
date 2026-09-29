import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from config import OUTPUT_DIR, DATA_PROCESSED, READMISSION_COST, INTERVENTION_COST, RISK_REDUCTION, TOP_PCT
from src.utils import save_metrics
from src import plot_style as ps

PER = 10000  # report everything per 10,000 discharges so it's easy to scale


def program_value(y, score, pct, cost=INTERVENTION_COST, rrr=RISK_REDUCTION, seed=42):
    # enrol the top pct of discharges by score in the care transition program
    rng = np.random.default_rng(seed)
    n = len(y)
    k = int(np.ceil(n * pct))
    order = np.lexsort((rng.random(n), -score))
    captured = y[order[:k]].sum()
    prevented = captured * rrr
    savings = prevented * READMISSION_COST
    spend = k * cost
    scale = PER / n
    return {
        "targeted": k * scale,
        "readmissions_reached": captured * scale,
        "readmissions_prevented": prevented * scale,
        "gross_savings": savings * scale,
        "program_cost": spend * scale,
        "net_savings": (savings - spend) * scale,
        "roi_pct": (savings - spend) / spend * 100 if spend else 0,
    }


def curve(y, score, cost=INTERVENTION_COST, rrr=RISK_REDUCTION):
    rows = []
    for pct in np.arange(1, 101) / 100:
        r = program_value(y, score, pct, cost, rrr)
        r["pct_targeted"] = round(pct * 100)
        rows.append(r)
    return pd.DataFrame(rows)


def chart_net_savings(curves, best_pct, best_net):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {"Gradient boosting": ps.BLUE, "LACE index": ps.ORANGE, "No targeting (random)": ps.AXIS}
    for name, c in curves.items():
        ax.plot(c["pct_targeted"], c["net_savings"] / 1000, color=colors[name], linewidth=2, label=name)
    ax.axhline(0, color=ps.INK_2, linewidth=0.8)
    ax.scatter([best_pct], [best_net / 1000], color=ps.BLUE, s=55, zorder=4, edgecolor=ps.SURFACE, linewidth=2)
    ax.annotate(f"best: target top {best_pct:.0f}% -> +${best_net / 1000:,.0f}K",
                xy=(best_pct, best_net / 1000), xytext=(14, 6), textcoords="offset points", fontsize=9, color=ps.INK)
    ax.set_xlabel("% of discharges enrolled in the program (highest risk first)")
    ax.set_ylabel("Net savings per 10,000 discharges ($K)")
    ax.set_xlim(0, 100)
    ax.set_ylim(top=best_net / 1000 + 180)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Model targeting makes the program pay; enrolling everyone loses money", pad=26)
    ps.subtitle(ax, f"Assumes ${INTERVENTION_COST} per patient, {RISK_REDUCTION:.0%} risk reduction, "
                    f"${READMISSION_COST:,} per readmission. Held-out test set.")
    ps.source_note(fig, "Cost sources: AHRQ HCUP (readmission cost), Coleman 2006 CTI trial & Jack 2009 Project RED (effect size)")
    return ps.save(fig, "12_net_savings_curve.png")


def chart_sensitivity(grid):
    table = grid.pivot(index="intervention_cost", columns="risk_reduction_pct", values="net_savings")
    best_pct = grid.pivot(index="intervention_cost", columns="risk_reduction_pct", values="best_pct_targeted")
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    lim = np.abs(table.values).max()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.imshow(table.values / 1000, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-lim / 1000, vmax=lim / 1000),
              aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            txt = f"${v / 1000:,.0f}K\ntop {best_pct.values[i, j]:.0f}%" if v > 0 else "no profitable\ntarget"
            color = "white" if abs(v) > lim * 0.55 else ps.INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color)
    ax.set_xticks(range(table.shape[1]), [f"{c}%" for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [f"${c}" for c in table.index])
    ax.set_xlabel("Program effect (relative drop in readmission risk)")
    ax.set_ylabel("Program cost per patient")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Sensitivity: best achievable net savings per 10,000 discharges", pad=26)
    ps.subtitle(ax, "Each cell re-optimises how many patients to enrol. Blue = saves money, red = loses money.")
    return ps.save(fig, "13_sensitivity_heatmap.png")


def main():
    ps.set_style()
    pred = pd.read_csv(DATA_PROCESSED / "test_predictions.csv")
    y = pred["readmit_30"].values
    rng = np.random.default_rng(0)
    scores = {
        "Gradient boosting": pred["p_hgb"].values,
        "LACE index": pred["lace_score"].values.astype(float),
        "No targeting (random)": rng.random(len(y)),
    }

    # the program pays off when P(readmit) * effect * cost_of_readmission > program cost
    break_even_risk = INTERVENTION_COST / (RISK_REDUCTION * READMISSION_COST)
    print(f"break-even risk = {break_even_risk:.1%} (enrol a patient only if their risk is above this)")

    curves = {name: curve(y, s) for name, s in scores.items()}
    all_curves = pd.concat([c.assign(strategy=n) for n, c in curves.items()])
    all_curves.round(2).to_csv(OUTPUT_DIR / "cost_benefit_curves.csv", index=False)

    model_curve = curves["Gradient boosting"]
    best = model_curve.loc[model_curve["net_savings"].idxmax()]

    # side by side at the same staffing level (top 20%) + the classic rules
    rows = []
    for name, s in scores.items():
        r = program_value(y, s, TOP_PCT)
        r["strategy"] = f"{name} - top {TOP_PCT:.0%}"
        rows.append(r)
    lace_rule_pct = (pred["lace_score"] >= 10).mean()
    r = program_value(y, scores["LACE index"], lace_rule_pct)
    r["strategy"] = f"LACE >= 10 rule ({lace_rule_pct:.0%} of stays)"
    rows.append(r)
    r = program_value(y, scores["Gradient boosting"], best["pct_targeted"] / 100)
    r["strategy"] = f"Gradient boosting - optimal top {best['pct_targeted']:.0f}%"
    rows.append(r)
    r = program_value(y, scores["No targeting (random)"], 1.0)
    r["strategy"] = "Enrol every discharge"
    rows.append(r)
    comparison = pd.DataFrame(rows)[["strategy", "targeted", "readmissions_reached", "readmissions_prevented",
                                     "gross_savings", "program_cost", "net_savings", "roi_pct"]].round(1)
    comparison.to_csv(OUTPUT_DIR / "cost_strategy_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # sensitivity: re-optimise the enrolment % for every cost / effect combo
    grid_rows = []
    for cost in [250, 500, 750, 1000]:
        for rrr in [0.10, 0.15, 0.20, 0.25, 0.30]:
            c = curve(y, scores["Gradient boosting"], cost, rrr)
            b = c.loc[c["net_savings"].idxmax()]
            grid_rows.append({"intervention_cost": cost, "risk_reduction_pct": int(rrr * 100),
                              "best_pct_targeted": b["pct_targeted"], "net_savings": round(b["net_savings"], 0)})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUTPUT_DIR / "cost_sensitivity_grid.csv", index=False)

    chart_net_savings(curves, best["pct_targeted"], best["net_savings"])
    chart_sensitivity(grid)

    at20 = program_value(y, scores["Gradient boosting"], TOP_PCT)
    lace20 = program_value(y, scores["LACE index"], TOP_PCT)
    everyone = program_value(y, scores["No targeting (random)"], 1.0)
    save_metrics("cost", {
        "assumptions": {"readmission_cost": READMISSION_COST, "intervention_cost": INTERVENTION_COST,
                        "risk_reduction": RISK_REDUCTION},
        "break_even_risk_pct": round(break_even_risk * 100, 2),
        "optimal_pct_targeted": float(best["pct_targeted"]),
        "optimal_net_savings_per_10k": round(best["net_savings"], 0),
        "optimal_prevented_per_10k": round(best["readmissions_prevented"], 1),
        "optimal_roi_pct": round(best["roi_pct"], 1),
        "top20_net_savings_per_10k": round(at20["net_savings"], 0),
        "top20_prevented_per_10k": round(at20["readmissions_prevented"], 1),
        "top20_roi_pct": round(at20["roi_pct"], 1),
        "lace_top20_net_savings_per_10k": round(lace20["net_savings"], 0),
        "enrol_everyone_net_per_10k": round(everyone["net_savings"], 0),
        "profitable_scenarios": int((grid["net_savings"] > 0).sum()),
        "total_scenarios": int(len(grid)),
    })
    print(f"optimal: enrol top {best['pct_targeted']:.0f}% -> net ${best['net_savings']:,.0f} per 10k discharges "
          f"({best['readmissions_prevented']:.0f} readmissions prevented)")


if __name__ == "__main__":
    main()
