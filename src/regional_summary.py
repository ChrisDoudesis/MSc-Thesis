# regional_summary.py
# Aggregate municipal EV charger estimates (rf/xgb_EV_area_distr.csv) into the
# five Danish regions (NUTS 2, cf. https://www.dst.dk/en/Statistik/dokumentation/nomenklaturer/nuts)
# with a selectable probability threshold.
#
# Usage:
#   python regional_summary.py                      # both models, tau = 0.90
#   python regional_summary.py --threshold 70       # tau = 0.70
#   python regional_summary.py --model rf --threshold 95
#
# Output: printed table + <model>_regional_p<threshold>.csv (for the map later).

import argparse
import pandas as pd

# --- Municipality (KOM) -> Region, per DST NUTS nomenclature ---------------
# Verify against the DST source before final use.
REGIONS = {
    "Hovedstaden": [
        101, 147, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 173, 175,
        183, 185, 187, 190, 201, 210, 217, 219, 223, 230, 240, 250, 260, 270,
        400,  # Bornholm
    ],
    "Sjælland": [
        253, 259, 265, 269, 306, 316, 320, 326, 329, 330, 336, 340, 350, 360,
        370, 376, 390,
    ],
    "Syddanmark": [
        410, 420, 430, 440, 450, 461, 479, 480, 482, 492, 510, 530, 540, 550,
        561, 563, 573, 575, 580, 607, 621, 630,
    ],
    "Midtjylland": [
        615, 657, 661, 665, 671, 706, 707, 710, 727, 730, 740, 741, 746, 751,
        756, 760, 766, 779, 791,
    ],
    "Nordjylland": [
        773, 787, 810, 813, 820, 825, 840, 846, 849, 851, 860,
    ],
}
KOM_TO_REGION = {kom: reg for reg, koms in REGIONS.items() for kom in koms}

VALID_THRESHOLDS = (50, 70, 80, 90, 95)


def regional_summary(csv_path: str, threshold: int = 90) -> pd.DataFrame:
    """Sum municipal estimates into regions for the selected threshold (50/70/80/90/95)."""
    if threshold not in VALID_THRESHOLDS:
        raise ValueError(f"threshold must be one of {VALID_THRESHOLDS}")
    col = f"n_evs_p{threshold}"

    df = pd.read_csv(csv_path)
    df["region"] = df["KOM"].map(KOM_TO_REGION)

    unmapped = df.loc[df["region"].isna(), "KOM"].tolist()
    if unmapped:
        raise ValueError(f"KOM codes without region mapping: {unmapped}")

    out = (
        df.groupby("region")
          .agg(n_municipalities=("KOM", "size"),
               n_meters=("n_meters", "sum"),
               n_evs=(col, "sum"))
          .reindex(REGIONS.keys())
    )
    out["penetration_pct"] = (out["n_evs"] / out["n_meters"] * 100).round(2)
    # meter-weighted mean of the annual mean probability, for reference
    w = df["n_meters"] * df["mean_yearly_ev_probability"]
    out["weighted_mean_prob"] = (
        w.groupby(df["region"]).sum() / df.groupby("region")["n_meters"].sum()
    ).reindex(REGIONS.keys()).round(4)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "xgb", "both"], default="both")
    ap.add_argument("--threshold", type=int, default=90, choices=VALID_THRESHOLDS)
    args = ap.parse_args()

    models = ["rf", "xgb"] if args.model == "both" else [args.model]
    for m in models:
        res = regional_summary(f"{m}_EV_area_distr.csv", args.threshold)
        print(f"\n=== {m.upper()} — regional estimates at P >= {args.threshold}% ===")
        print(res.to_string())
        out_csv = f"{m}_regional_p{args.threshold}.csv"
        res.to_csv(out_csv)
        print(f"saved {out_csv}")
