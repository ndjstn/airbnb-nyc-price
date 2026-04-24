# Room Type First, Geography Second: Airbnb NYC 2019 Price Modelling

A gradient-boosted price model on the 2019 NYC Airbnb snapshot reaches R-squared 0.619 on log price and MAE under 50 USD per night. The finding that pays off the modelling work is not the number. It is the feature ranking. Sixty-two percent of the model's decision weight goes to room type indicators. The first geographic feature — distance to Times Square — sits at 14.8 percent. On this dataset, a listing is first of all a product category and only secondarily a location.

## Key results

| Model | R-squared on log price | MAE (USD) |
| --- | ---: | ---: |
| Baseline Ridge | 0.5207 | 54.77 |
| Spatial Ridge | 0.5585 | 53.33 |
| Gradient Boosting | 0.6190 | 49.79 |

Adding five landmark-distance features to a ridge baseline buys 3.8 percentage points of R-squared. Moving to gradient boosting buys another 6. The model is fit on log(price+1) with a 75/25 random split.

The top three feature importances from the gradient booster: `room_type_Private room` at 0.520, `dist_times_square` at 0.148, `room_type_Shared room` at 0.100. The raw data agrees. Entire home / apt median is 160 USD per night, private room is 70 USD, shared room is 45 USD, before any geography is considered.

## What is in this repo

`src/run_analysis.py` is a single end-to-end script that loads the Kaggle CSV, filters to a sensible price band, engineers Haversine distances to five NYC landmarks, fits three models at increasing capacity, and writes the figures and tables including an interactive Folium heatmap. `figures/nyc-price-map.html` is the interactive heatmap overlaid with markers for the ten most-expensive high-volume neighborhoods. `outputs/` holds the model comparison table and the feature-importance JSON.

`REPORT.md` is a long-form written analysis covering the data, the feature engineering, the three models, the finding, and the honest limits on what a 2019 snapshot can tell a 2026 reader about a market that has materially changed under Local Law 18.

## How to reproduce

The dataset is the `AB_NYC_2019.csv` file from the Kaggle Dgomonov dataset. Download from <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data> and place it at `data/AB_NYC_2019.csv` relative to the repo root.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_analysis.py --data data/AB_NYC_2019.csv --figures figures --outputs outputs
```

Open `figures/nyc-price-map.html` in a browser to view the interactive map. Total runtime is under a minute on a modern CPU.

## Further reading

The full write-up with narrative and the interactive map is on my site: <https://ndjstn.github.io/posts/airbnb-nyc-price-room-type-geography/>.

## License

MIT. See [LICENSE](LICENSE).
