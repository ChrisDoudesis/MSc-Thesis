# make_municipality_map.py
# Choropleth of estimated EV charger penetration per municipality (tau = 0.90).
# One figure per model (shared colour scale across models for comparability).
# Requirements: pip install geopandas matplotlib requests
# Boundaries: official DAWA / Dataforsyningen API (public), joined on KOM code.

import io
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

RF_CSV  = "rf_EV_area_distr.csv"
XGB_CSV = "xgb_EV_area_distr.csv"
GEO_URL = "https://api.dataforsyningen.dk/kommuner?format=geojson"

# --- load model outputs -------------------------------------------------
def load(path, label):
    df = pd.read_csv(path)
    df["pen90"] = df["n_evs_p90"] / df["n_meters"] * 100.0
    return df[["KOM", "pen90"]].rename(columns={"pen90": label})

data = load(RF_CSV, "RF").merge(load(XGB_CSV, "XGB"), on="KOM")

# --- load municipal boundaries ------------------------------------------
# DAWA 'kode' is a zero-padded string ("0101"); convert to int to match KOM.
r = requests.get(GEO_URL, timeout=120)
r.raise_for_status()
gdf = gpd.read_file(io.BytesIO(r.content))
gdf["KOM"] = gdf["kode"].astype(int)
gdf = gdf.merge(data, on="KOM", how="left")
# lighter file / faster rendering (tolerance in degrees; boundaries stay visually intact)
gdf["geometry"] = gdf.geometry.simplify(0.002)

# --- plot: one figure per model, shared colour scale ---------------------
vmax = gdf[["RF", "XGB"]].max().max()
for col, title, suffix in [("RF", "Random Forest", "rf"), ("XGB", "XGBoost", "xgb")]:
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(column=col, ax=ax, cmap="viridis", vmin=0, vmax=vmax,
             edgecolor="white", linewidth=0.3,
             legend=True,
             legend_kwds={"label": "Estimated EV charger penetration (% of meters), $\\tau=0.90$",
                          "shrink": 0.6},
             missing_kwds={"color": "lightgrey"})
    ax.set_title(title)
    ax.set_axis_off()

    plt.tight_layout()
    out_pdf = f"municipality_choropleth_{suffix}.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {out_pdf}")
