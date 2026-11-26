# Urban-Congestion-Patterns-under-High-Electric-Vehicle-Penetration-A-Case-Study-of-10-U.S.-Cities
Research code accompanying the paper "Urban Congestion Patterns under High Electric Vehicle Penetration: A Case Study of 10 U.S. Cities".

## 1. Code overview

| Script | Description |
| --- | --- |
| `cal_grid.py` | Constructs income-based spatial grids from census tract data. |
| `cal_e_g_od.py` | Allocates electric vehicle (EV) demand by income group and generates separate origin–destination (OD) matrices for gasoline vehicles (GVs) and EVs. |
| `traffic_assignment.py` | Performs multiclass traffic assignment using AequilibraE and calculates average travel time for different EV penetration scenarios. |
| `moran_I.py` | Calculates Global Moran's I to evaluate the spatial autocorrelation of changes in link-level congestion. |
| `k_means.py` | Classifies cities according to their travel-time trends using K-means clustering. |

## 2. Workflow

Run the scripts from the project root directory containing the city data folders. The default configuration covers 10 cities and 21 EV penetration scenarios, from 0% to 100% in increments of 5%.

```text
cal_grid.py
    └── Income grids (.gpkg)
          ↓
cal_e_g_od.py
    └── GV and EV OD matrices (.csv)
          ↓
traffic_assignment.py
    ├── Link-level traffic assignment results (.csv)
    │     └── moran_I.py → Spatial autocorrelation results (.csv)
    └── Average travel time by city and scenario (.csv)
          └── k_means.py → City clusters (.csv) and visualization (.png)
```

### Step 1. Generate income grids

```bash
python cal_grid.py
```

Input: `{city_folder}/grid_shp/{city_folder}.shp`.

Output: `{city_folder}/{city_name}_income_grid.gpkg`.

### Step 2. Generate GV and EV OD matrices

```bash
python cal_e_g_od.py
```

Inputs: the income grid from Step 1 and the node and OD tables in `{city_folder}/01_Input_data/`.

Outputs for each scenario `i`:

```text
{city_folder}/od_matrix/{i}/
├── {city_name}_od_matrix_g.csv
├── {city_name}_od_matrix_e.csv
└── {city_name}_od_split_by_ev_rate.csv
```

### Step 3. Perform traffic assignment

```bash
python traffic_assignment.py
```

Inputs: the OD matrices from Step 2, the original OD table, and the road network shapefile at `{city_folder}/03_AequilibraE_results/{city_name}.shp`.

With `save_record=True`, the script generates:

```text
{city_folder}/04_my_AequilibraE_results/{i}/
├── network_{i}.csv
└── assignment_result_{i}.csv
```

It also saves `recordtime_gamma_e={gamma_e}.csv` in the project root directory. Each row corresponds to a penetration scenario, each column to a city, and each value to average travel time in minutes.

### Step 4. Analyze the assignment results

The following scripts can be run independently after Step 3:

```bash
python moran_I.py
python k_means.py
```

- `moran_I.py` reads link-level assignment results for the 0% and 100% EV scenarios and requires the corresponding network topology file at `{city_folder}/03_AequilibraE_results/network.csv`. It generates `moran_i_results.csv`.
- `k_means.py` reads `recordtime_gamma_e=1.0.csv` and generates `kmeans_city_clusters_result.csv` and `kmeans_city_trend_clusters.png`.

## 3. Key parameters

### 3.1. `cal_grid.py` — Income grid construction

The script groups census tracts according to their median household income, using Census B19001 income brackets.

| Parameter | Description |
| --- | --- |
| `B19001_BINS` | Household income brackets used to estimate median income. |
| `groups` | Ten income groups ordered from highest to lowest income. `grid_id = 0` represents the highest-income group. |
| `folders` and the final city-index loop | Determine which cities are processed. |

### 3.2. `cal_e_g_od.py` — EV demand allocation

EV demand is allocated across income grids and the original OD demand is split into GV and EV demand according to the income grid of each origin node.

The linear income score is defined as:

$$
 y = kx + b, \qquad x = \frac{I_{\mathrm{grid}}}{I_{\mathrm{city}}}
$$

where $I_{\mathrm{grid}}$ is the representative income of the grid and $I_{\mathrm{city}}$ is the citywide median household income.

| Parameter | Description |
| --- | --- |
| `e_rate` | Citywide EV penetration rate, calculated as `i / 20`. |
| `func` | Income-based allocation function: `"linear"` or `"sigmoid"`. |
| `args=[k, b]` | Slope and intercept of the linear income score. A positive `k` favors higher-income groups, subject to the score bounds and demand constraints. |
| `beta` | Controls the sensitivity to income differences when using the sigmoid function (`args=[beta]`). |
| `min_score`, `max_score` | Bounds of the linear income score (0 and 1 by default). |
| `open_ended_multiplier` | Multiplier for estimating representative income in the open-ended highest-income group (3.0 by default in the linear function). |

The current scenario configuration is:

```python
g_od_df, e_od_df, od_data, e_rate_list = build_gas_ev_od_matrices(
    e_rate=i / 20,
    grid=grid,
    folder=folder,
    func="linear",
    args=[0.0, 1.0],
    save=True,
)
```

With `k = 0` and `b = 1`, all income grids have the same score, so EV demand is allocated proportionally to grid OD demand without income-based differentiation. Total EV demand is constrained to the target citywide penetration rate, and grid-level EV demand cannot exceed the original grid OD demand.

### 3.3. `traffic_assignment.py` — Multiclass traffic assignment

The script assigns GV and EV traffic to a shared road network using the biconjugate Frank–Wolfe (BFW) algorithm in AequilibraE.

| Parameter | Description |
| --- | --- |
| `gamma_g`, `gamma_e` | Value-of-time parameters for GVs and EVs, respectively, passed to `TrafficClass.set_vot()`. |
| `k_g_list`, `k_e_list` | City-specific per-distance fixed-cost coefficients for GVs and EVs. Link fixed costs are calculated as link length multiplied by the corresponding coefficient. |
| `alpha_precondition`, `beta_precondition` | City-specific parameters of the BPR volume–delay function. |
| `road_1/2/3_capacity_precondition` | Road capacities for the three road classes. |
| `road_1/2/3_speed_precondition` | Free-flow speeds (km/h) for the three road classes. |
| `multiplier` | OD demand scaling factor in `generate_aem_from_matrix()`; default: 0.6. |
| `city_ids`, `i_range` | Selected cities and penetration scenarios. |
| `max_workers` | Number of parallel worker processes; default: 6. |
| `save_record` | Enables saving link-level results and the average travel-time table. |

The BPR function is defined as:

$$
 t = t_0 \left[1 + \alpha \left(\frac{v}{c}\right)^\beta\right]
$$

where $t_0$ is the free-flow travel time, $v$ is traffic volume, and $c$ is road capacity. The assignment uses a maximum of 500 iterations and a target relative gap of 0.001.

The current main configuration is:

```python
fun(
    city_ids=[0, 2, 3, 5, 8, 9, 12, 13, 18, 19],
    i_range=list(range(0, 21)),
    gamma_g=1.0,
    gamma_e=1.0,
    save_record=True,
)
```

### 3.4. `moran_I.py` — Spatial autocorrelation

The script calculates Global Moran's I using the absolute relative change in link-level volume-to-capacity ratio (VOC) between the 0% and 100% EV penetration scenarios:

$$
 x_l = \left|\frac{VOC_{l,100\%}-VOC_{l,0\%}}{VOC_{l,0\%}+10^{-12}}\right|
$$

| Parameter | Description |
| --- | --- |
| `value_col` | Preferred congestion indicator; default: `"VOC_max"`. Other VOC fields are used as fallbacks if necessary. |
| `city_ids` | Cities included in the spatial analysis. |
| `W` | Binary link adjacency matrix. Two links are adjacent if they share at least one endpoint node. |

### 3.5. `k_means.py` — City clustering

The script uses seven features derived from the average travel-time curve: total reduction, early-stage reduction, middle-stage reduction, late-stage reduction, early reduction share, late reduction share, and the degree of non-monotonicity.

| Parameter | Description |
| --- | --- |
| `path` | Average travel-time CSV file; default: `recordtime_gamma_e=1.0.csv`. |
| `required_i` | Reference scenarios `[0, 6, 14, 20]`, corresponding to 0%, 30%, 70%, and 100% EV penetration. |
| `k` | Number of clusters; default: 3. The current Type I–III label mapping is configured for three clusters. |
| `feature_indices` | Selected trend features; the current configuration uses all seven. |
| `feature_weights` | Weights applied to the selected features after standardization. |
| `random_state` | Random seed; default: 42. |

The current feature weights are `[1.0, 0.8, 1.0, 1.2, 0.6, 1.2, 2.0]`. PCA is used to visualize the clustering results in two dimensions when more than two features are selected.

## 4. Requirements and usage notes

The main Python dependencies are NumPy, pandas, GeoPandas, Shapely, SciPy, Matplotlib, OSMnx, AequilibraE, and scikit-learn. The optional `adjustText` package improves label placement in the clustering figure.

- Keep the city folder names and input file structure consistent with the scripts.
- The traffic-assignment script is referred to as `traffic_assignment.py` in this README; use the actual filename if it differs.
- The assignment script currently uses Windows-style separators for its per-scenario output files. On other operating systems, use platform-independent paths before running it.
- If `gamma_e` changes, update the input filename in `k_means.py`. Preserve the link-level outputs separately if comparing multiple assignment configurations, since their filenames do not include the gamma values.

