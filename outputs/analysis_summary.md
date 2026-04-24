# Airbnb NYC 2019 price analysis summary

Filtered rows (price in [1, 2000] USD): 48,798.
Median price: $105. Mean price: $145.53.

## Model comparison (test set)

| Model | R^2 on log(price) | MAE in USD |
|---|---:|---:|
| baseline_ridge | 0.5207 | $54.77 |
| spatial_ridge | 0.5585 | $53.33 |
| gradient_boosting | 0.6190 | $49.79 |

Baseline Ridge uses basic listing columns and borough/room-type dummies. Spatial Ridge adds the five landmark-distance features. Gradient boosting uses the full spatial feature set.