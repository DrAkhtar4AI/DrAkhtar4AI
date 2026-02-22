#!/usr/bin/env python3
"""Map forest types in Pakistan using NASA PACE observations + supervised learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import earthaccess
import geopandas as gpd
import numpy as np
import rasterio
import rioxarray  # noqa: F401 (registers .rio accessor)
import xarray as xr
from rasterio.transform import from_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Pakistan bounding box (lon_min, lat_min, lon_max, lat_max)
PAKISTAN_BBOX = (60.8, 23.5, 77.9, 37.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--training-samples", required=True, type=Path)
    parser.add_argument("--out-raster", required=True, type=Path)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def search_and_download_pace(start_date: str, end_date: str, download_dir: Path) -> list[Path]:
    earthaccess.login(persist=True)

    # Collection name may evolve; update if Earthdata changes naming.
    results = earthaccess.search_data(
        short_name="PACE_OCI_L2_AOP_NRT",
        temporal=(start_date, end_date),
        bounding_box=PAKISTAN_BBOX,
        count=20,
    )

    if not results:
        raise RuntimeError("No PACE granules found for the selected dates and Pakistan bbox.")

    downloaded = earthaccess.download(results, local_path=str(download_dir))
    return [Path(p) for p in downloaded]


def open_and_stack_features(granule_paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []

    for granule in granule_paths:
        ds = xr.open_dataset(granule)

        # Try common OCI variable names used in atmospheric optics products.
        candidates = [
            "Rrs_443",
            "Rrs_555",
            "Rrs_670",
            "aot_865",
        ]
        existing = [v for v in candidates if v in ds.data_vars]

        if not existing:
            continue

        subset = ds[existing]
        datasets.append(subset)

    if not datasets:
        raise RuntimeError("Could not find expected reflectance/aerosol variables in downloaded PACE files.")

    merged = xr.concat(datasets, dim="time")

    # Reduce temporal noise: median composite.
    composite = merged.median(dim="time", skipna=True)
    return composite


def sample_training_pixels(
    composite: xr.Dataset,
    samples_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if "forest_type" not in samples_gdf.columns:
        raise ValueError("Training data must include a 'forest_type' column.")

    feature_names = list(composite.data_vars)
    X, y = [], []

    for _, row in samples_gdf.iterrows():
        point = row.geometry.centroid
        lon, lat = point.x, point.y

        pixel = composite.sel(lon=lon, lat=lat, method="nearest")
        values = [float(pixel[f].values) for f in feature_names]

        if any(np.isnan(values)):
            continue

        X.append(values)
        y.append(row["forest_type"])

    if not X:
        raise RuntimeError("No valid training samples intersected composite data.")

    return np.asarray(X), np.asarray(y), feature_names


def classify_raster(
    composite: xr.Dataset,
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    out_raster: Path,
) -> None:
    feature_names = list(composite.data_vars)

    lon = composite["lon"].values
    lat = composite["lat"].values
    width = lon.size
    height = lat.size

    stack = np.stack([composite[f].values for f in feature_names], axis=0)
    stack_2d = stack.reshape(len(feature_names), -1).T

    valid_mask = np.all(np.isfinite(stack_2d), axis=1)

    pred = np.full(stack_2d.shape[0], fill_value=0, dtype=np.uint16)
    pred[valid_mask] = model.predict(stack_2d[valid_mask]).astype(np.uint16)
    pred_img = pred.reshape(height, width)

    transform = from_bounds(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()), width, height)

    out_raster.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_raster,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=pred_img.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(pred_img, 1)

    class_legend = out_raster.with_suffix(".classes.txt")
    lines = ["0: nodata"] + [f"{i + 1}: {name}" for i, name in enumerate(label_encoder.classes_)]
    class_legend.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    downloads_dir = Path("data/raw_pace")
    downloads_dir.mkdir(parents=True, exist_ok=True)

    granules = search_and_download_pace(args.start_date, args.end_date, downloads_dir)
    composite = open_and_stack_features(granules)

    samples = gpd.read_file(args.training_samples)
    X, y, feature_names = sample_training_pixels(composite, samples)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(1, len(label_encoder.classes_) + 1),
            target_names=label_encoder.classes_,
            digits=3,
        )
    )

    print("Feature importance:")
    for fname, importance in sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {fname}: {importance:.4f}")

    classify_raster(composite, model, label_encoder, args.out_raster)
    print(f"Saved classified map: {args.out_raster}")


if __name__ == "__main__":
    main()
