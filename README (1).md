# City Traffic Assignment and EV Demand Analysis

## 1. What each file does

  -----------------------------------------------------------------------------------------
  File                        Purpose                 Main output
  --------------------------- ----------------------- -------------------------------------
  `cal_grid.py`               Builds income-based     `{city_name}_income_grid.gpkg`
                              spatial grids from      
                              Census tract polygons   
                              and B19001              
                              household-income data.  
                              Each grid is assigned   
                              an income range and the 
                              citywide median income. 

  `cal_e_g_od.py`             Matches tract nodes to  `od_matrix/{rate}/` containing
                              the income grids,       gasoline and EV OD matrices plus the
                              calculates OD demand    split OD table
                              for each grid,          
                              allocates the citywide  
                              EV share across grids,  
                              and splits OD demand    
                              into gasoline and EV    
                              matrices.               

  `traffic_assignment.py`\*   Runs parallel           `04_my_AequilibraE_results/{rate}/`
                              AequilibraE traffic     and
                              assignments for         `recordtime_gamma_e={gamma_e}.csv`
                              different EV            
                              penetration levels. It  
                              calculates link-level   
                              assignment results and  
                              average traffic time.   

  `moran_I.py`                Calculates Global       `moran_i_results.csv`
                              Moran's I for the       
                              spatial pattern of      
                              relative link-level VOC 
                              changes between EV      
                              penetration levels 0    
                              and 1. Link adjacency   
                              is defined by shared    
                              endpoint nodes.         

  `k_means.py`                Extracts traffic-time   `kmeans_city_trend_clusters.png` and
                              trend features from the `kmeans_city_clusters_result.csv`
                              record-time CSV,        
                              clusters cities into    
                              three groups using      
                              KMeans, and plots the   
                              result.                 
  -----------------------------------------------------------------------------------------

\* The traffic-assignment code was uploaded as `粘贴的 Markdown (1).md`.
Save the Python code inside it as `traffic_assignment.py` before running
it.

## 2. Recommended execution order

Run the scripts from the project root directory, where the city folders
are located.

### Step 1 --- Build the income grids

``` bash
python cal_grid.py
```

For each selected city, this reads:

``` text
{city_folder}/grid_shp/{city_folder}.shp
```

It creates:

``` text
{city_folder}/{city_name}_income_grid.gpkg
```

The grids are ordered from the highest-income group to the lowest-income
group (`grid_id = 0` is the highest-income group).

### Step 2 --- Match nodes and generate gasoline/EV OD matrices

``` bash
python cal_e_g_od.py
```

This step reads the income grid generated in Step 1, together with:

``` text
{city_folder}/01_Input_data/{city_name}_node.csv
{city_folder}/01_Input_data/{city_name}_od.csv
```

It matches tract nodes to income-grid polygons, calculates each grid's
outgoing OD demand, allocates the EV share, and splits the OD table into
gasoline and EV demand.

For each city and EV penetration level, it creates:

``` text
{city_folder}/od_matrix/{rate}/{city_name}_od_matrix_g.csv
{city_folder}/od_matrix/{rate}/{city_name}_od_matrix_e.csv
{city_folder}/od_matrix/{rate}/{city_name}_od_split_by_ev_rate.csv
```

Here, `rate` runs from `0` to `20`, representing EV penetration levels
from `0.00` to `1.00` in increments of `0.05`.

### Step 3 --- Run traffic assignment

Save the traffic-assignment code as `traffic_assignment.py`, then run:

``` bash
python traffic_assignment.py
```

The script reads the gasoline and EV OD matrices from Step 2 and
performs traffic assignment for each selected city and EV penetration
level.

With `save_record=True`, it saves link-level results under:

``` text
{city_folder}/04_my_AequilibraE_results/{rate}/
```

The main outputs include:

``` text
network_{rate}.csv
assignment_result_{rate}.csv
recordtime_gamma_e={gamma_e}.csv
```

The record-time CSV stores the average traffic time for each city at
each EV penetration level.

### Step 4 --- Run the two downstream analyses

After traffic assignment is complete, the following two scripts can be
run independently.

**A. Spatial autocorrelation of VOC changes**

``` bash
python moran_I.py
```

This compares the assignment results at EV penetration `0` and `20` and
saves:

``` text
moran_i_results.csv
```

It also requires the original network topology file:

``` text
{city_folder}/03_AequilibraE_results/network.csv
```

**B. City trend clustering**

``` bash
python k_means.py
```

This reads `recordtime_gamma_e=1.0.csv` and saves:

``` text
kmeans_city_trend_clusters.png
kmeans_city_clusters_result.csv
```

If a different `gamma_e` is used in Step 3, update the input filename in
`k_means.py` accordingly.

## 3. Key parameters by file

### `cal_grid.py` --- Income-grid construction

-   `folders`: the list of city folders.
-   `B19001_BINS`: the official B19001 household-income brackets.
-   `groups`: combines the official brackets into 10 income groups. The
    order is from highest income to lowest income, so `grid_id = 0` is
    the highest-income group.
-   `city_ids` in the final loop: selects which cities are processed.

The script estimates each city's median income by aggregating the B19001
household counts across its census tracts. The open-ended top income
bracket (`$200,000+`) is treated conservatively as `$200,000` when
estimating the median.

### `cal_e_g_od.py` --- Income-based EV allocation and OD splitting

-   `e_rate`: citywide EV share. The current loop uses `i / 20`,
    covering `0.00`--`1.00` in 21 steps.

-   `func`: selects the allocation function. The current active setting
    is `"linear"`.

-   `args=[k, b]`: parameters for the linear income score:

    \[ x =
    `\frac{\text{representative grid income}}{\text{citywide median income}}`{=tex}
    \]

    \[ `\text{score}`{=tex} = kx + b \]

-   `min_score` and `max_score`: lower and upper bounds applied to the
    linear score. Their defaults are `0.0` and `1.0`.

-   `open_ended_multiplier`: representative-income multiplier for the
    open-ended `$200,000+` group. The linear function defaults to `3.0`.

-   Alternative: the sigmoid allocation function uses `beta` (default
    `2.5`) to control how strongly income differences affect the score.

**Important: the current main loop passes `func="linear"` and
`args=[0.0, 1.0]`.** This makes `score = 1` for every grid, so the
current run does not differentiate EV allocation weights by income. The
linear function's own defaults (`k=0.15`, `b=0.10`) are overridden by
the values passed in the main loop.

### `traffic_assignment.py` --- AequilibraE traffic assignment

-   `city_ids`: cities included in the assignment.
-   `i_range`: EV penetration scenarios. The current main call uses `0`
    through `20`, corresponding to `e_rate = i / 20`.
-   `gamma_g` and `gamma_e`: values passed to `set_vot()` for the
    gasoline and EV traffic classes. The current main call sets both to
    `1.0`.
-   `alpha_precondition` and `beta_precondition`: city-specific
    parameters of the BPR volume-delay function.
-   `road_1/2/3_capacity_precondition` and
    `road_1/2/3_speed_precondition`: city-specific road-class capacities
    and speeds.
-   `k_g_list` and `k_e_list`: city-specific coefficients used to
    calculate gasoline and EV link fixed costs
    (`link length × coefficient`).
-   `max_workers`: number of parallel worker processes; default `6`.
-   Assignment convergence settings: `max_iter = 500` and
    `rgap_target = 0.001`.
-   OD matrix scaling: `generate_aem_from_matrix()` defaults to
    `multiplier=0.6`.

The current main call is configured with `gamma_g=1.0`, `gamma_e=1.0`,
and `save_record=True`.

### `moran_I.py` --- Global Moran's I

-   `city_ids`: cities to analyse.

-   `value_col`: VOC field to use. The default is `"VOC_max"`; the
    script also checks alternative field names such as `"VOC_AB"` if the
    preferred field is unavailable.

-   Comparison scenarios: the script reads assignment results from EV
    penetration folders `0` and `20`.

-   Link adjacency: two links are neighbours if they share at least one
    endpoint node.

-   Change variable:

    \[ x_i =
    `\left`{=tex}\|`\frac{VOC_{i,\;EV=100\%}-VOC_{i,\;EV=0\%}}{VOC_{i,\;EV=0\%}+10^{-12}}`{=tex}`\right`{=tex}\|
    \]

    Moran's I is then calculated for the spatial distribution of this
    relative VOC change.

### `k_means.py` --- City trend clustering

-   Input file: `recordtime_gamma_e=1.0.csv` by default.

-   `required_i = [0, 6, 14, 20]`: scenario rows used as the start,
    early, middle, and final reference points.

-   `k = 3`: number of city clusters.

-   `feature_indices`: selects which trend features are used. The
    current setting uses all seven features (`0`--`6`).

-   `feature_weights`: weights applied after standardization. The
    current setting is:

    ``` python
    [1.0, 0.8, 1.0, 1.2, 0.6, 1.2, 2.0]
    ```

-   `random_state = 42`: makes the KMeans initialization reproducible.

The seven features are:

  --------------------------------------------------------------------------
                         Index Feature                 Meaning
  ---------------------------- ----------------------- ---------------------
                             0 `total_drop`            Total relative
                                                       traffic-time
                                                       reduction from
                                                       scenario 0 to 20

                             1 `early_drop`            Reduction from
                                                       scenario 0 to 6

                             2 `mid_drop`              Reduction from
                                                       scenario 6 to 14

                             3 `late_drop`             Reduction from
                                                       scenario 14 to 20

                             4 `early_share`           Early-stage reduction
                                                       as a share of the
                                                       total reduction

                             5 `late_share`            Late-stage reduction
                                                       as a share of the
                                                       total reduction

                             6 `monotonic_violation`   Sum of upward changes
                                                       in the traffic-time
                                                       series, normalized by
                                                       the initial value
  --------------------------------------------------------------------------

Before KMeans, the selected features are standardized. `feature_weights`
are then applied to control their relative influence on clustering.
