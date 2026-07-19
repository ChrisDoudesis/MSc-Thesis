# make_regional_map.py
# Choropleth ("heatmap") of estimated EV charger penetration for the
# five Danish regions, at a selectable probability threshold.
# One figure per model (shared colour scale across models for comparability).
# Boundaries: official DAWA API (public). Uses regional_summary.py (same folder).
#
# Usage:
#   python make_regional_map.py                          # both models, tau=0.90
#   python make_regional_map.py --model xgb --threshold 70
#
# Requirements: pip install geopandas matplotlib requests

import io
import argparse
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from regional_summary import regional_summary, VALID_THRESHOLDS

GEO_URL = "https://api.dataforsyningen.dk/regioner?format=geojson"

# DAWA region names -> names used in regional_summary.REGIONS
DAWA_TO_REGION = {
    "Nordjylland": "Nordjylland",
    "Midtjylland": "Midtjylland",
    "Syddanmark": "Syddanmark",
    "Hovedstaden": "Hovedstaden",
    "Sjælland": "Sjælland",
}


def load_boundaries() -> gpd.GeoDataFrame:
    r = requests.get(GEO_URL, timeout=120)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content))
    gdf["region"] = gdf["navn"].str.replace("Region ", "", regex=False).map(DAWA_TO_REGION)
    gdf["geometry"] = gdf.geometry.simplify(0.005)  # lighter rendering
    return gdf


def plot(models, threshold, cmap="PuBuGn"):
    gdf = load_boundaries()

    # shared colour scale across models so the separate maps stay comparable
    tables = {m: regional_summary(f"{m}_EV_area_distr.csv", threshold) for m in models}
    vmax = max(t["penetration_pct"].max() for t in tables.values())

    for m in models:
        fig, ax = plt.subplots(figsize=(8, 8))
        t = tables[m]
        g = gdf.merge(t, left_on="region", right_index=True)
        g.plot(column="penetration_pct", ax=ax, cmap=cmap, vmin=0, vmax=vmax,
               edgecolor="white", linewidth=0.8,
               legend=True,
               legend_kwds={"label": f"EV charger penetration (% of meters), "
                                     f"$\\tau={threshold/100:.2f}$", "shrink": 0.6})
        # annotate each region with name and value; bold white with a thin black
        # outline stays readable on any map shade and on the white background
        for _, row in g.iterrows():
            c = row.geometry.representative_point()
            ax.annotate(f"{row['region']}\n{row['penetration_pct']:.1f}%",
                        xy=(c.x, c.y), ha="center", fontsize=9,
                        color="white", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        ax.set_title({"rf": "EV Chargers Distribution per Region - Random Forest",
                      "xgb": "EV Chargers Distribution per Region - XGBoost"}[m])
        ax.set_axis_off()

        plt.tight_layout()
        out_pdf = f"regional_map_{m}_p{threshold}.pdf"
        plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"saved {out_pdf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "xgb", "both"], default="both")
    ap.add_argument("--threshold", type=int, default=90, choices=VALID_THRESHOLDS)
    ap.add_argument("--cmap", default="PuBuGn")
    args = ap.parse_args()
    models = ["rf", "xgb"] if args.model == "both" else [args.model]
    plot(models, args.threshold, cmap=args.cmap)
