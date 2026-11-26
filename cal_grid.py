from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd


folders = [
    "01_San_Francisco",
    "02_Seattle",
    "03_Portland",
    "04_Las_Vegas",
    "05_Chicago",
    "06_New_Orleans",
    "07_Austin",
    "08_Minneapolis",
    "09_Dallas",
    "10_Milwaukee",
    "11_New_York",
    "12_Washington",
    "13_Boston",
    "14_Philadelphia",
    "15_Pittsburgh",
    "16_Miami",
    "17_Atlanta",
    "18_Phoenix",
    "19_Denver",
    "20_Honolulu"
]


# ============================================================
# Official B19001 income brackets
# ============================================================

B19001_BINS = {
    "B19001002": (0, 10000),
    "B19001003": (10000, 15000),
    "B19001004": (15000, 20000),
    "B19001005": (20000, 25000),
    "B19001006": (25000, 30000),
    "B19001007": (30000, 35000),
    "B19001008": (35000, 40000),
    "B19001009": (40000, 45000),
    "B19001010": (45000, 50000),
    "B19001011": (50000, 60000),
    "B19001012": (60000, 75000),
    "B19001013": (75000, 100000),
    "B19001014": (100000, 125000),
    "B19001015": (125000, 150000),
    "B19001016": (150000, 200000),
    "B19001017": (200000, np.inf),
}

BIN_CODES = list(B19001_BINS.keys())
TOTAL_COL = "B19001001"
def income_codes_to_range(income_codes):
    """
    Convert ["B19001014", "B19001015"] to [100000, 149999].
    Convert ["B19001017"] to [200000, None].
    """
    lows = []
    highs = []

    for code in income_codes:
        low, high = B19001_BINS[code]
        lows.append(low)
        highs.append(high)

    income_min = int(min(lows))
    income_max_raw = max(highs)

    if np.isinf(income_max_raw):
        income_max = None
    else:
        # The high value in B19001_BINS is an exclusive upper bound.
        # For example, 150000 means the actual maximum value is 149999.
        income_max = int(income_max_raw - 1)

    return [income_min, income_max]

# ============================================================
# Important:
# grid_id = 0 corresponds to the highest-income group.
# Therefore, groups must be ordered from highest to lowest income.
# ============================================================

groups = [
    ["B19001017"],                                # 200k or more
    ["B19001016"],                                # 150-199,999
    ["B19001014", "B19001015"],                  # 100-149,999
    ["B19001013"],                                # 75-99,999
    ["B19001011", "B19001012"],                  # 50-74,999
    ["B19001008", "B19001009", "B19001010"],    # 35-49,999
    ["B19001006", "B19001007"],                  # 25-34,999
    ["B19001004", "B19001005"],                  # 15-24,999
    ["B19001003"],                                # 10-14,999
    ["B19001002"],                                # Less than 10k
]
def estimate_median_income_from_counts(counts, total=None):
    """
    Estimate the overall median income from household counts in the B19001 income brackets.

    counts: Series indexed by B19001002 through B19001017.
    total: optional total number of households.
    """
    counts = pd.to_numeric(counts, errors="coerce").fillna(0.0).clip(lower=0.0)

    if total is None or not np.isfinite(total) or total <= 0:
        total = counts.sum()

    if not np.isfinite(total) or total <= 0 or counts.sum() <= 0:
        return np.nan

    half = total / 2.0
    cum = counts.cumsum().values

    idx = int(np.searchsorted(cum, half, side="left"))
    idx = min(idx, len(BIN_CODES) - 1)

    median_code = BIN_CODES[idx]
    bin_count = float(counts.iloc[idx])

    low, high = B19001_BINS[median_code]

    cum_before = 0.0 if idx == 0 else float(cum[idx - 1])

    if bin_count <= 0:
        return float(low)

    if np.isinf(high):
        # The highest bracket is 200k+ and has no upper bound, so use 200000 as a conservative estimate.
        return float(low)

    ratio_inside_bin = (half - cum_before) / bin_count
    ratio_inside_bin = float(np.clip(ratio_inside_bin, 0.0, 1.0))

    median_income = low + ratio_inside_bin * (high - low)

    return float(median_income)

CODE_TO_GRID_ID = {}
for grid_id, codes in enumerate(groups):
    for code in codes:
        CODE_TO_GRID_ID[code] = grid_id


def get_median_income_bin(row):
    """
    Determine the B19001 income bracket containing the median income of each census tract based on household counts in each bracket.
    Return the corresponding B190010xx code.
    """
    counts = pd.to_numeric(row[BIN_CODES], errors="coerce").fillna(0.0).clip(lower=0.0)

    total = pd.to_numeric(pd.Series([row.get(TOTAL_COL, np.nan)]), errors="coerce").iloc[0]

    if not np.isfinite(total) or total <= 0:
        total = counts.sum()

    if not np.isfinite(total) or total <= 0 or counts.sum() <= 0:
        return None

    half = total / 2.0
    cum = counts.cumsum().values

    idx = int(np.searchsorted(cum, half, side="left"))
    idx = min(idx, len(BIN_CODES) - 1)

    return BIN_CODES[idx]


def build_income_grid_for_city(city_path):
    city_path = Path(city_path)
    city_folder_name = city_path.name
    city_name = city_folder_name[3:] if len(city_folder_name) > 3 else city_folder_name

    shp_path = city_path / "grid_shp" / f"{city_folder_name}.shp"

    if not shp_path.exists():
        raise FileNotFoundError(f"No shp: {shp_path}")

    gdf = gpd.read_file(shp_path)

    missing_cols = [c for c in [TOTAL_COL] + BIN_CODES if c not in gdf.columns]
    if missing_cols:
        raise ValueError(f"{city_folder_name} lack: {missing_cols}")

    # --------------------------------------------------------
    # Keep only census-tract polygons.
    # Records with geoid=14000US... represent census tracts.
    # Records with geoid=16000US... represent city totals and should be excluded.
    # --------------------------------------------------------
    if "geoid" in gdf.columns:
        tract_gdf = gdf[gdf["geoid"].astype(str).str.startswith("14000US")].copy()
    else:
        tract_gdf = gdf[gdf.geometry.geom_type == "Polygon"].copy()

    tract_gdf = tract_gdf[
        tract_gdf.geometry.notna() & (~tract_gdf.geometry.is_empty)
    ].copy()

    if len(tract_gdf) == 0:
        raise ValueError(f"{city_folder_name} no polygon。")

    # Fix invalid geometries.
    try:
        from shapely.validation import make_valid
        tract_gdf["geometry"] = tract_gdf.geometry.apply(make_valid)
    except Exception:
        tract_gdf["geometry"] = tract_gdf.buffer(0)

    # --------------------------------------------------------
    # Determine the median-income bracket for each census tract.
    # --------------------------------------------------------
    tract_gdf["median_bin_code"] = tract_gdf.apply(get_median_income_bin, axis=1)

    tract_gdf = tract_gdf.dropna(subset=["median_bin_code"]).copy()

    tract_gdf["grid_id"] = tract_gdf["median_bin_code"].map(CODE_TO_GRID_ID)

    # --------------------------------------------------------
    # Calculate the citywide median income.
    # Calculate it by aggregating B19001 income-bracket household counts across all census tracts.
    # --------------------------------------------------------
    city_counts = tract_gdf[BIN_CODES].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).clip(lower=0.0).sum(axis=0)

    city_total = pd.to_numeric(
        tract_gdf[TOTAL_COL], errors="coerce"
    ).fillna(0.0).clip(lower=0.0).sum()

    city_midincome = estimate_median_income_from_counts(
        counts=city_counts,
        total=city_total,
    )

    print(f"   city_midincome = {city_midincome:.2f}")

    tract_gdf = tract_gdf.dropna(subset=["grid_id"]).copy()
    tract_gdf["grid_id"] = tract_gdf["grid_id"].astype(int)

    # --------------------------------------------------------
    # Aggregate polygons by income group.
    # Output only the following fields: grid_id, income, and polygon.
    # --------------------------------------------------------
    rows = []

    for grid_id, income_codes in enumerate(groups):
        sub = tract_gdf[tract_gdf["grid_id"] == grid_id].copy()

        # Skip income groups with no census tracts to avoid None or NaN values.
        if len(sub) == 0:
            continue

        merged_polygon = sub.geometry.union_all()

        income_range = income_codes_to_range(income_codes)

        rows.append({
            "grid_id": grid_id,
            "income": json.dumps(income_range, ensure_ascii=False),
            "midincome": float(city_midincome),
            "polygon": merged_polygon,
        })

    out_gdf = gpd.GeoDataFrame(
        rows,
        geometry="polygon",
        crs=gdf.crs,
    )

    out_gdf = out_gdf.sort_values("grid_id").reset_index(drop=True)

    # Fix the merged geometries.
    try:
        from shapely.validation import make_valid
        out_gdf["polygon"] = out_gdf.geometry.apply(make_valid)
    except Exception:
        out_gdf["polygon"] = out_gdf.geometry.buffer(0)

    out_gdf = out_gdf[["grid_id", "income", "midincome", "polygon"]]

    # --------------------------------------------------------
    # Save the output.
    # Remove any existing output file to avoid retaining fields from a previous version.
    # --------------------------------------------------------
    output_path = city_path / f"{city_name}_income_grid.gpkg"

    if output_path.exists():
        output_path.unlink()

    out_gdf.to_file(
        output_path,
        layer="income_grid",
        driver="GPKG",
    )
    float(city_midincome)
    print(f"✅ saved: {output_path}")
    print(out_gdf)

    return out_gdf


# Run for the selected cities.
for i in [0, 2, 3, 5, 8, 9, 12, 13, 18, 19]:
    city_path = folders[i]
    income_grid_gdf = build_income_grid_for_city(city_path)


