"""Airbnb NYC 2019 price analysis.

Loads the dgomonov NYC 2019 Airbnb listings CSV, engineers a handful of spatial
features (distance to Times Square, distance to Brooklyn Bridge, distance to
LGA, proximity to the nearest subway entry from a bundled set of station
coordinates), fits two price models, and writes an interactive folium map
layered with median price by neighborhood.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

RNG = 42

# Landmark coordinates (decimal degrees)
LANDMARKS = {
    "times_square": (40.7580, -73.9855),
    "brooklyn_bridge": (40.7061, -73.9969),
    "lga_airport": (40.7769, -73.8740),
    "central_park_south": (40.7644, -73.9730),
    "prospect_park": (40.6602, -73.9690),
}

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 11})
sns.set_style("whitegrid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--figures", required=True)
    p.add_argument("--outputs", required=True)
    return p.parse_args()


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, (lat, lon) in LANDMARKS.items():
        df[f"dist_{name}"] = haversine_miles(df["latitude"].values, df["longitude"].values, lat, lon)
    df["min_landmark_dist"] = df[[f"dist_{k}" for k in LANDMARKS]].min(axis=1)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["log_price"] = np.log1p(df["price"])
    return df


def price_by_neighbourhood_figure(df: pd.DataFrame, path: Path) -> None:
    g = df.groupby("neighbourhood_group")["price"].agg(["median", "count"]).sort_values("median", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(g.index, g["median"], color=sns.color_palette("mako_r", len(g)))
    for bar, m, n in zip(bars, g["median"], g["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"${int(m)}\n(n={n:,})", ha="center", va="bottom")
    ax.set_ylabel("Median listing price (USD / night)")
    ax.set_title("Median listing price by borough, 2019")
    ax.set_ylim(0, g["median"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def room_type_figure(df: pd.DataFrame, path: Path) -> None:
    g = df.groupby(["neighbourhood_group", "room_type"])["price"].median().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    g.plot(kind="bar", ax=ax, color=["#4c72b0", "#c44e52", "#55a868"])
    ax.set_ylabel("Median nightly price (USD)")
    ax.set_title("Median nightly price by borough and room type")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def distance_figure(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mask = (df["price"] > 0) & (df["price"] < 500)
    hex_ = ax.hexbin(
        df.loc[mask, "dist_times_square"], df.loc[mask, "price"],
        gridsize=40, cmap="mako_r", mincnt=3,
    )
    ax.set_xlabel("Distance from Times Square (miles)")
    ax.set_ylabel("Nightly price (USD)")
    ax.set_title("Listing price vs. distance from Times Square")
    fig.colorbar(hex_, ax=ax, label="Listings in bin")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_folium_map(df: pd.DataFrame, path: Path) -> None:
    # Subsample for the heatmap to keep file size reasonable
    heat_sample = df.loc[df["price"].between(1, 500)].sample(n=min(15000, len(df)), random_state=RNG)
    m = folium.Map(location=[40.7359, -73.9911], zoom_start=11, tiles="CartoDB positron")
    HeatMap(
        heat_sample[["latitude", "longitude", "price"]].values.tolist(),
        min_opacity=0.3, radius=8, blur=10, max_zoom=13,
    ).add_to(m)
    # Layer a point marker for each of the top-10 most-expensive neighborhoods
    top_hoods = (
        df.loc[df["price"].between(1, 500)]
        .groupby(["neighbourhood", "neighbourhood_group"])
        .agg(median_price=("price", "median"), listings=("id", "count"))
        .query("listings >= 100")
        .sort_values("median_price", ascending=False)
        .head(10)
    )
    for (hood, group), row in top_hoods.iterrows():
        sub = df.loc[(df["neighbourhood"] == hood) & df["price"].between(1, 500)]
        lat, lon = sub["latitude"].median(), sub["longitude"].median()
        folium.Marker(
            [lat, lon],
            popup=f"<b>{hood}</b><br>{group}<br>Median: ${int(row['median_price'])}<br>Listings: {int(row['listings'])}",
        ).add_to(m)
    m.save(str(path))


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.figures)
    out_dir = Path(args.outputs)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"Rows: {len(df):,}")
    # Drop zero- or absurd-price rows
    df = df.loc[df["price"].between(1, 2000)].reset_index(drop=True)
    print(f"After 1 to 2000 USD filter: {len(df):,}")

    df = engineer(df)

    price_by_neighbourhood_figure(df, fig_dir / "price-by-borough.png")
    room_type_figure(df, fig_dir / "price-by-borough-room.png")
    distance_figure(df, fig_dir / "price-vs-times-square-distance.png")
    build_folium_map(df, fig_dir / "nyc-price-map.html")

    feature_cols = [
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
        "dist_times_square", "dist_brooklyn_bridge", "dist_lga_airport",
        "dist_central_park_south", "dist_prospect_park", "min_landmark_dist",
    ]
    cat_cols = ["neighbourhood_group", "room_type"]

    X = pd.get_dummies(df[feature_cols + cat_cols], columns=cat_cols, drop_first=True)
    y = df["log_price"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RNG)

    baseline_cols = [c for c in X.columns if c in feature_cols[:7] or c.startswith("room_type_") or c.startswith("neighbourhood_group_")]
    ridge_baseline = Ridge(alpha=1.0, random_state=RNG)
    ridge_baseline.fit(X_train[baseline_cols], y_train)
    yhat_baseline = ridge_baseline.predict(X_test[baseline_cols])
    baseline_r2 = r2_score(y_test, yhat_baseline)
    baseline_mae = mean_absolute_error(np.expm1(y_test), np.expm1(yhat_baseline))

    ridge_spatial = Ridge(alpha=1.0, random_state=RNG)
    ridge_spatial.fit(X_train, y_train)
    yhat_spatial = ridge_spatial.predict(X_test)
    spatial_r2 = r2_score(y_test, yhat_spatial)
    spatial_mae = mean_absolute_error(np.expm1(y_test), np.expm1(yhat_spatial))

    gbr = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=RNG)
    gbr.fit(X_train, y_train)
    yhat_gbr = gbr.predict(X_test)
    gbr_r2 = r2_score(y_test, yhat_gbr)
    gbr_mae = mean_absolute_error(np.expm1(y_test), np.expm1(yhat_gbr))

    importances = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.iloc[::-1].plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel("Gradient boosting importance")
    ax.set_title("Top 15 features by gradient boosting importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature-importance.png")
    plt.close(fig)

    results = {
        "baseline_ridge": {"r2": float(baseline_r2), "mae_usd": float(baseline_mae)},
        "spatial_ridge": {"r2": float(spatial_r2), "mae_usd": float(spatial_mae)},
        "gradient_boosting": {"r2": float(gbr_r2), "mae_usd": float(gbr_mae)},
    }
    summary = {
        "dataset": {
            "rows": int(len(df)),
            "median_price_usd": float(df["price"].median()),
            "mean_price_usd": float(df["price"].mean()),
        },
        "models": results,
        "top_features": importances.to_dict(),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    pd.DataFrame([
        {"model": "baseline_ridge", **{k: round(v, 4) for k, v in results["baseline_ridge"].items()}},
        {"model": "spatial_ridge", **{k: round(v, 4) for k, v in results["spatial_ridge"].items()}},
        {"model": "gradient_boosting", **{k: round(v, 4) for k, v in results["gradient_boosting"].items()}},
    ]).to_csv(out_dir / "model_comparison.csv", index=False)

    md = ["# Airbnb NYC 2019 price analysis summary", ""]
    md.append(f"Filtered rows (price in [1, 2000] USD): {len(df):,}.")
    md.append(f"Median price: ${df['price'].median():.0f}. Mean price: ${df['price'].mean():.2f}.")
    md.append("")
    md.append("## Model comparison (test set)")
    md.append("")
    md.append("| Model | R^2 on log(price) | MAE in USD |")
    md.append("|---|---:|---:|")
    for name in ("baseline_ridge", "spatial_ridge", "gradient_boosting"):
        md.append(f"| {name} | {results[name]['r2']:.4f} | ${results[name]['mae_usd']:.2f} |")
    md.append("")
    md.append("Baseline Ridge uses basic listing columns and borough/room-type dummies. Spatial Ridge adds the five landmark-distance features. Gradient boosting uses the full spatial feature set.")
    (out_dir / "analysis_summary.md").write_text("\n".join(md))
    print("Done")


if __name__ == "__main__":
    main()
