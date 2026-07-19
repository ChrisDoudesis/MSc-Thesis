# plot_type_distributions.py
# Two-panel bar chart: estimated EV charger penetration per dwelling/consumption
# type (DEF_KODE) and per heating type (VARME), RF vs XGBoost, selectable threshold.
#
# Usage:
#   python plot_type_distributions.py                 # tau = 0.90
#   python plot_type_distributions.py --threshold 70
#
# Expects in the working directory:
#   rf_EV_hus_type.csv,  xgb_EV_hus_type.csv   (DEF_KODE files)
#   rf_EV_varme_type.csv, xgb_EV_varme_type.csv (VARME files)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEF_LABELS = {
    100: "Dwellings (unspecified)",
    111: "Apartment, no electric heating",
    112: "Apartment, electric heating",
    113: "Apartment, heat pump",
    121: "Single-family house, no electric heating",
    122: "Single-family house, electric heating",
    123: "Single-family house, heat pump",
    131: "Summer house, no electric heating",
    132: "Summer house, electric heating",
    133: "Summer house, heat pump",
    134: "Allotment garden",
}
VARME_LABELS = {
    0: "Not stated",
    1: "None",
    2: "Other heating (district/gas/oil)",
    3: "Electric heating",
    4: "Heat pump",
    5: "Mixed",
}

MIN_METERS = 2000   # categories below this are dropped (footnoted in the caption)


def load(prefix, code_col, labels, threshold):
    col = f"n_evs_p{threshold}"
    out = {}
    for m in ("rf", "xgb"):
        df = pd.read_csv(f"{m}_EV_{prefix}.csv")
        df["label"] = df[code_col].map(labels)
        df["pen"] = df[col] / df["n_meters"] * 100
        out[m] = df.set_index("label")[["pen", "n_meters"]]
    merged = out["rf"].join(out["xgb"], lsuffix="_rf", rsuffix="_xgb")
    merged = merged[merged["n_meters_rf"] >= MIN_METERS]
    return merged.sort_values("pen_rf")


def main(threshold):
    hus = load("hus_type", "DEF_KODE", DEF_LABELS, threshold)
    varme = load("varme_type", "VARME", VARME_LABELS, threshold)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title in zip(axes, [hus, varme],
                               ["Dwelling / consumption type (DEF\\_KODE)",
                                "Heating type (VARME)"]):
        y = np.arange(len(data))
        ax.barh(y + 0.2, data["pen_rf"], height=0.4, label="Random Forest")
        ax.barh(y - 0.2, data["pen_xgb"], height=0.4, label="XGBoost")
        ax.set_yticks(y, data.index)
        ax.set_xlabel(f"EV charger penetration (% of meters), "
                      f"$\\tau={threshold/100:.2f}$")
        ax.set_title(title)
        # annotate meter counts for context
        for yi, (_, row) in zip(y, data.iterrows()):
            ax.annotate(f"n={int(row['n_meters_rf']):,}",
                        xy=(max(row['pen_rf'], row['pen_xgb']), yi),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7, color="grey")
    axes[0].legend(loc="lower right")
    plt.tight_layout()
    out = f"type_distributions_p{threshold}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=90,
                    choices=[50, 70, 80, 90, 95])
    args = ap.parse_args()
    main(args.threshold)
