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


def get_city_name(folder):
    folder = Path(folder)
    name = folder.name
    return name[3:] if len(name) > 3 and name[:2].isdigit() else name


def parse_income_value(x):
    """
    income may be provided as:
        "[100000, 149999]"
        [100000, 149999]
        "[200000, null]"
        [200000, None]
    """
    if isinstance(x, str):
        return json.loads(x)
    if isinstance(x, (list, tuple)):
        return list(x)
    raise ValueError(f"error income: {x}")


def income_range_to_representative_income(income_range):
    """
    Convert [a, b] to a representative income.
    For an open-ended interval such as [200000, None], use 250000 as a conservative representative value.
    """
    a, b = income_range

    a = float(a)

    if b is None or pd.isna(b):
        return a * 3.0

    b = float(b)
    return 0.5 * (a + b)


def read_income_grid(folder):
    """
    Read:
        ./folder/{city_name}_income_grid.gpkg

    Return a GeoDataFrame containing:
        grid_id
        income       Python list: [a, b] or [a, None]
        midincome
        polygon
    """
    folder = Path(folder)
    city_name = get_city_name(folder)

    grid_path = folder / f"{city_name}_income_grid.gpkg"

    if not grid_path.exists():
        raise FileNotFoundError(f"no income grid: {grid_path}")

    grid = gpd.read_file(grid_path)

    if "polygon" in grid.columns:
        grid = grid.set_geometry("polygon")
    else:
        grid = grid.rename_geometry("polygon")

    required_cols = ["grid_id", "income", "midincome"]
    missing = [c for c in required_cols if c not in grid.columns]
    if missing:
        raise ValueError(f"{grid_path} lack: {missing}")

    grid = grid[["grid_id", "income", "midincome", "polygon"]].copy()

    grid["grid_id"] = grid["grid_id"].astype(int)
    grid["income"] = grid["income"].apply(parse_income_value)
    grid["midincome"] = pd.to_numeric(grid["midincome"], errors="coerce")

    if grid["midincome"].isna().any():
        raise ValueError("income grid have NaN midincome。")

    grid = grid.sort_values("grid_id").reset_index(drop=True)
    return grid


def read_tract_nodes(folder):
    """
    Read:
        ./folder/01_Input_data/{city_name}_node.csv

    Keep only Node_ID values where Tract_Node == 1.
    """
    folder = Path(folder)
    city_name = get_city_name(folder)

    node_path = folder / "01_Input_data" / f"{city_name}_node.csv"

    if not node_path.exists():
        raise FileNotFoundError(f"No node: {node_path}")

    nodes = pd.read_csv(node_path)

    required_cols = ["Node_ID", "Lon", "Lat", "Tract_Node"]
    missing = [c for c in required_cols if c not in nodes.columns]
    if missing:
        raise ValueError(f"{node_path} lack: {missing}")

    nodes = nodes[nodes["Tract_Node"] == 1].copy()

    nodes["Node_ID"] = nodes["Node_ID"].astype(int)
    nodes["Lon"] = pd.to_numeric(nodes["Lon"], errors="coerce")
    nodes["Lat"] = pd.to_numeric(nodes["Lat"], errors="coerce")

    nodes = nodes.dropna(subset=["Lon", "Lat"]).copy()

    node_gdf = gpd.GeoDataFrame(
        nodes,
        geometry=gpd.points_from_xy(nodes["Lon"], nodes["Lat"]),
        crs="EPSG:4326",
    )

    return node_gdf


def read_od_data(folder):
    """
    Read:
        ./folder/01_Input_data/{city_name}_od.csv
    """
    folder = Path(folder)
    city_name = get_city_name(folder)

    od_path = folder / "01_Input_data" / f"{city_name}_od.csv"

    if not od_path.exists():
        raise FileNotFoundError(f"No OD: {od_path}")

    od = pd.read_csv(od_path)

    required_cols = ["O_ID", "D_ID", "OD_Number"]
    missing = [c for c in required_cols if c not in od.columns]
    if missing:
        raise ValueError(f"{od_path} lack: {missing}")

    od["O_ID"] = od["O_ID"].astype(int)
    od["D_ID"] = od["D_ID"].astype(int)
    od["OD_Number"] = pd.to_numeric(od["OD_Number"], errors="coerce").fillna(0.0)

    return od

def plot_grid_node_match_debug(
    grid,
    joined,
    folder,
    filename=None,
    max_node_labels=2000,
):
    """
    Plot the spatial matches between income-grid polygons and Tract_Node points.

    Label each polygon with its grid_id.
    Label each node with its matched grid_id.
    Label unmatched nodes as NA:Node_ID.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    folder = Path(folder)
    city_name = get_city_name(folder)

    debug_dir = folder / "debug_grid_node_match"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{city_name}_grid_node_match_debug.png"

    save_path = debug_dir / filename

    # --------------------------------------------------------
    # The grid geometry column may be named polygon rather than geometry.
    # --------------------------------------------------------
    grid_plot = grid.copy()

    if "polygon" in grid_plot.columns:
        grid_plot = grid_plot.set_geometry("polygon")

    grid_geom_col = grid_plot.geometry.name

    # --------------------------------------------------------
    # The geometry in joined represents node points.
    # --------------------------------------------------------
    node_plot = joined.copy()

    if "geometry" not in node_plot.columns:
        raise ValueError("joined have no geometry")

    node_plot = gpd.GeoDataFrame(
        node_plot,
        geometry="geometry",
        crs=grid_plot.crs,
    )

    node_geom_col = node_plot.geometry.name

    node_plot["grid_label"] = node_plot["grid_id"].apply(
        lambda x: "NA" if pd.isna(x) else str(int(x))
    )

    unmatched_plot = node_plot[node_plot["grid_id"].isna()].copy()
    matched_plot = node_plot[~node_plot["grid_id"].isna()].copy()

    fig, ax = plt.subplots(figsize=(14, 14))

    # --------------------------------------------------------
    # Plot polygon boundaries.
    # --------------------------------------------------------
    grid_plot.boundary.plot(
        ax=ax,
        linewidth=0.8,
        color="black",
    )

    # --------------------------------------------------------
    # Plot matched nodes.
    # --------------------------------------------------------
    if len(matched_plot) > 0:
        matched_plot.plot(
            ax=ax,
            markersize=8,
            alpha=0.7,
            color="blue",
        )

    # --------------------------------------------------------
    # Plot unmatched nodes.
    # --------------------------------------------------------
    if len(unmatched_plot) > 0:
        unmatched_plot.plot(
            ax=ax,
            markersize=40,
            alpha=0.9,
            color="red",
        )

    # --------------------------------------------------------
    # Label grid_id values inside polygons.
    # --------------------------------------------------------
    for _, row in grid_plot.iterrows():
        geom = row[grid_geom_col]

        if geom is None or geom.is_empty:
            continue

        p = geom.representative_point()

        ax.text(
            p.x,
            p.y,
            str(int(row["grid_id"])),
            fontsize=16,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )

    # --------------------------------------------------------
    # Label each node with its matched grid_id.
    # If there are too many nodes, label unmatched nodes first, followed by a subset of matched nodes.
    # --------------------------------------------------------
    if len(node_plot) > max_node_labels:
        label_nodes = pd.concat(
            [
                unmatched_plot,
                matched_plot.head(max(0, max_node_labels - len(unmatched_plot))),
            ],
            axis=0,
        )
    else:
        label_nodes = node_plot

    for _, row in label_nodes.iterrows():
        geom = row[node_geom_col]

        if geom is None or geom.is_empty:
            continue

        x = geom.x
        y = geom.y

        if pd.isna(row["grid_id"]):
            label = f"NA:{int(row['Node_ID'])}"
            color = "red"
        else:
            label = str(int(row["grid_id"]))
            color = "blue"

        ax.text(
            x,
            y,
            label,
            fontsize=7,
            ha="left",
            va="bottom",
            color=color,
        )

    ax.set_title(
        f"{city_name} income grid / tract node spatial match\n"
        f"nodes={len(node_plot)}, unmatched={len(unmatched_plot)}",
        fontsize=14,
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"✅ grid-node match debug figure saved: {save_path}")

    return save_path


def match_nodes_to_income_grid(folder, grid=None, duplicate_policy="error"):
    """
    1. Read the income grid.
    2. Read nodes where Tract_Node == 1.
    3. Match nodes to income polygons using Lon/Lat.
    4. Use the OD table to retain only nodes that appear as O_ID and have outgoing demand > 0.
    5. Calculate Grid_OD_demand for each income grid.

    Return grid with the following fields:
        grid_id
        income
        midincome
        polygon
        Node_ID
        Grid_OD_demand
    """
    folder = Path(folder)

    if grid is None:
        grid = read_income_grid(folder)

    nodes = read_tract_nodes(folder)
    od = read_od_data(folder)

    if grid.crs is None:
        raise ValueError("income grid have no CRS")

    nodes = nodes.to_crs(grid.crs)

    # covered_by includes boundary points and is more robust than within.
    try:
        joined = gpd.sjoin(
            nodes[["Node_ID", "geometry"]],
            grid[["grid_id", "polygon"]],
            how="left",
            predicate="covered_by",
        )
    except Exception:
        joined = gpd.sjoin(
            nodes[["Node_ID", "geometry"]],
            grid[["grid_id", "polygon"]],
            how="left",
            predicate="intersects",
        )

    # Check for nodes that were not matched to any grid.

    city_name = get_city_name(folder)

    unmatched = joined[joined["grid_id"].isna()]["Node_ID"].astype(int).tolist()

    if len(unmatched) > 0:
        print("The following Node_ID values with Tract_Node == 1 are not contained in any income-grid polygon:")
        print(unmatched[:200])

        if len(unmatched) > 200:
            print(f"... and {len(unmatched) - 200} more not shown")

        # --------------------------------------------------------
        # City-specific manual assignments for San Francisco.
        # When folder = "01_San_Francisco",
        # get_city_name(folder) returns "San_Francisco".
        # --------------------------------------------------------
        if city_name in ["San_Francisco", "Portland", "Las_Vegas", "New_Orleans", "Dallas", "Milwaukee", "Boston", "Philadelphia", "Denver", "Honolulu"]:
            manual_node_to_grid = []
            if city_name in ["San_Francisco"]:
                manual_node_to_grid = {
                    10000001: 1,
                    10000003: 3,
                    10000020: 2,
                }
            elif city_name in ["Portland"]:
                manual_node_to_grid = {
                    10000006: 2,
                    10000019: 2,
                    10000020: 1,
                    10000033: 1,
                }
            elif city_name in ["Las_Vegas"]:
                manual_node_to_grid = {
                    10000000: 5,
                    10000007: 2,
                    10000008: 2,
                    10000022: 2,
                    10000025: 5,
                    10000026: 3,
                    10000027: 3,
                    10000028: 3,
                    10000048: 5,
                    10000075: 3,
                    10000082: 6,
                    10000083: 4,
                    10000098: 4,
                    10000106: 4,
                    10000113: 4,
                    10000115: 4,
                    10000130: 3,
                    10000131: 3,
                    10000150: 5,
                }
            elif city_name in ["New_Orleans"]:
                manual_node_to_grid = {
                    10000005: 5,
                    10000006: 1,
                    10000013: 8,
                    10000087: 4,
                    10000159: 5,
                    10000178: 4,
                    10000179: 1,
                    10000180: 6,
                    10000181: 1,
                    10000182: 3,
                    10000183: 2,
                    10000184: 2,

                }
            elif city_name in ["Dallas"]:
                manual_node_to_grid = {
                    10000007: 5,
                    10000013: 0,
                    10000015: 4,
                    10000016: 3,
                    10000022: 2,
                    10000034: 4,
                    10000035: 5,
                    10000036: 3,
                    10000038: 2,
                    10000040: 5,
                    10000041: 4,
                    10000046: 4,
                    10000057: 4,
                    10000059: 4,
                    10000066: 4,
                    10000073: 3,
                    10000075: 3,
                    10000091: 4,
                    10000313: 0,
                }
            elif city_name in ["Milwaukee"]:
                manual_node_to_grid = {
                    10000020: 5,
                    10000035: 5,
                    10000079: 4,
                    10000081: 3,
                    10000104: 4,
                    10000106: 4,
                    10000107: 4,
                    10000111: 4,
                    10000134: 4,
                    10000136: 4,
                    10000139: 4,
                    10000141: 4,
                    10000147: 4,
                    10000148: 4,
                    10000149: 4,
                    10000154: 4,
                    10000174: 3,
                    10000178: 3,
                    10000186: 4,
                    10000187: 4,
                    10000196: 5,
                    10000201: 3,
                    10000215: 5,
                    10000232: 3,
                    10000233: 4,
                }
            elif city_name in ["Boston"]:
                manual_node_to_grid = {
                    10000002: 2,
                    10000003: 2,
                    10000004: 2,
                    10000005: 5,
                    10000006: 1,
                    10000007: 5,
                    10000008: 5,
                    10000009: 3,
                    10000010: 3,
                    10000011: 3,
                    10000012: 2,
                    10000013: 1,
                    10000014: 2,
                    10000016: 4,
                    10000017: 3,
                    10000019: 0,
                    10000029: 4,
                    10000031: 1,
                    10000125: 2,
                    10000177: 3,
                }
            elif city_name in ["Boston"]:
                manual_node_to_grid = {
                    10000002: 2,
                    10000003: 2,
                    10000004: 2,
                    10000005: 5,
                    10000006: 1,
                    10000007: 5,
                    10000008: 5,
                    10000009: 3,
                    10000010: 3,
                    10000011: 3,
                    10000012: 2,
                    10000013: 1,
                    10000014: 2,
                    10000016: 4,
                    10000017: 3,
                    10000019: 0,
                    10000029: 4,
                    10000031: 1,
                    10000125: 2,
                    10000177: 3,
                }
            elif city_name in ["Philadelphia"]:
                manual_node_to_grid = {
                    10000001: 5 ,
                    10000013: 4,
                    10000032: 4,
                    10000039: 3,
                    10000051: 5,
                    10000081: 5,
                    10000082: 5,
                    10000107: 5,
                    10000138: 3,
                    10000139: 4,
                    10000171: 5,
                    10000378: 3,
                    10000379: 4,
                    10000380: 5,
                    10000381: 5,
                    10000382: 4,
                    10000383: 4,
                    10000384: 3,
                    10000385: 3,
                    10000386: 4,
                    10000387: 2,
                    10000388: 2,
                }
            elif city_name in ["Denver"]:
                manual_node_to_grid = {
                    10000000: 2,
                    10000001: 2,
                    10000042: 3,
                    10000053: 4,
                    10000088: 3,
                    10000143: 3,
                    10000144: 4,
                    10000145: 3,
                    10000146: 3,
                    10000147: 4,
                    10000148: 1,
                    10000149: 3,
                    10000150: 0,
                    10000151: 0,
                    10000152: 3,
                    10000153: 3,
                    10000154: 2,
                    10000155: 2,
                    10000156: 2,
                    10000157: 4,
                    10000158: 2,
                    10000159: 2,
                    10000160: 2,
                    10000161: 2,
                    10000162: 2,
                    10000163: 3,
                    10000164: 3,
                    10000165: 2,
                    10000167: 4,
                    10000168: 4,
                    10000169: 2,
                    10000170: 2,
                    10000171: 3,
                    10000172: 4,
                    10000173: 4,
                    10000174: 4,
                }
            elif city_name in ["Honolulu"]:
                manual_node_to_grid = {
                    10000012: 4,
                    10000013: 3,
                    10000014: 2,
                    10000017: 2,
                    10000018: 1,
                    10000019: 1,
                    10000031: 2,
                    10000032: 3,
                    10000033: 3,
                    10000037: 2,
                    10000055: 2,
                    10000057: 2,
                    10000058: 2,
                    10000059: 2,
                    10000060: 2,
                    10000061: 2,
                    10000062: 2,
                    10000115: 2,
                    10000116: 2,
                }





            unmatched_set = set(unmatched)
            manual_set = set(manual_node_to_grid.keys())

            can_manual_assign = unmatched_set.issubset(manual_set)

            if can_manual_assign:
                for node_id, manual_grid_id in manual_node_to_grid.items():
                    mask = joined["Node_ID"].astype(int) == int(node_id)

                    if mask.any():
                        joined.loc[mask, "grid_id"] = int(manual_grid_id)
                        print(
                            f"Manually assigned Node_ID={node_id} -> grid_id={manual_grid_id}"
                        )

                print("Unmatched nodes in San Francisco have been assigned manually.")

                # ------------------------------------------------
                # Recheck the matches after manual assignment.
                # ------------------------------------------------
                unmatched_after_manual = (
                    joined[joined["grid_id"].isna()]["Node_ID"]
                    .astype(int)
                    .tolist()
                )

                if len(unmatched_after_manual) > 0:
                    plot_grid_node_match_debug(
                        grid=grid,
                        joined=joined,
                        folder=folder,
                        filename=f"{city_name}_grid_node_match_debug_after_manual_failed.png",
                    )

                    print("The following Node_ID values remain unmatched after manual assignment:")
                    print(unmatched_after_manual[:200])

                    if len(unmatched_after_manual) > 200:
                        print(f"... and {len(unmatched_after_manual) - 200} more not shown")

                    raise ValueError(
                        f"{city_name}手动分配后仍有 {len(unmatched_after_manual)} 个 node 没有匹配到 income grid。"
                    )

            else:
                # An unmatched node was found that is not included in the manual assignment list.
                unknown_unmatched = sorted(list(unmatched_set - manual_set))

                plot_grid_node_match_debug(
                    grid=grid,
                    joined=joined,
                    folder=folder,
                    filename=f"{city_name}_grid_node_match_debug_need_manual.png",
                )

                print(f"{city_name} has unmatched Node_ID values that are not included in the manual assignment list:")
                print(unknown_unmatched[:200])

                if len(unknown_unmatched) > 200:
                    print(f"... and {len(unknown_unmatched) - 200} more not shown")

                raise ValueError(
                    "请根据可视化结果手动规划这些 Node_ID 的 grid_id 分配。"
                )

        else:
            # --------------------------------------------------------
            # For cities other than San Francisco, plot unmatched nodes before raising an error.
            # --------------------------------------------------------
            plot_grid_node_match_debug(
                grid=grid,
                joined=joined,
                folder=folder,
                filename=f"{city_name}_grid_node_match_debug_need_manual.png",
            )

            raise ValueError(
                f"{city_name} 存在 {len(unmatched)} 个 node 没有匹配到 income grid。"
                " 请根据可视化结果手动规划这些 Node_ID 的 grid_id 分配。"
            )

    else:
        # If there are no unmatched nodes, plot the matching results for verification.
        plot_grid_node_match_debug(
            grid=grid,
            joined=joined,
            folder=folder,
            filename=f"{city_name}_grid_node_match_debug.png",
        )


    # Check whether any node is matched to multiple grids.
    dup_counts = joined.groupby("Node_ID")["grid_id"].nunique()
    duplicated_nodes = dup_counts[dup_counts > 1].index.astype(int).tolist()

    if len(duplicated_nodes) > 0:
        if duplicate_policy == "error":
            print("The following Node_ID values are matched to multiple income grids. Check for overlapping polygons or points on polygon boundaries:")
            print(duplicated_nodes[:200])
            raise ValueError(f"存在 {len(duplicated_nodes)} 个 node 匹配多个 grid。")
        elif duplicate_policy == "first":
            joined = joined.sort_values(["Node_ID", "grid_id"]).drop_duplicates("Node_ID", keep="first")
        else:
            raise ValueError(f"未知 duplicate_policy: {duplicate_policy}")
    else:
        joined = joined.drop_duplicates("Node_ID", keep="first")

    joined["grid_id"] = joined["grid_id"].astype(int)

    # Total OD demand for each O_ID.
    origin_demand = (
        od.groupby("O_ID", as_index=True)["OD_Number"]
        .sum()
        .astype(float)
    )

    # Keep only nodes that appear as O_ID and have a total OD_Number greater than 0.
    joined["Origin_OD_demand"] = joined["Node_ID"].map(origin_demand).fillna(0.0)
    joined = joined[joined["Origin_OD_demand"] > 0].copy()

    node_list_by_grid = (
        joined.groupby("grid_id")["Node_ID"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )

    demand_by_grid = (
        joined.groupby("grid_id")["Origin_OD_demand"]
        .sum()
        .to_dict()
    )

    grid = grid.copy()
    grid["Node_ID"] = grid["grid_id"].map(node_list_by_grid).apply(
        lambda x: x if isinstance(x, list) else []
    )
    grid["Grid_OD_demand"] = grid["grid_id"].map(demand_by_grid).fillna(0.0)

    grid = grid[["grid_id", "income", "midincome", "polygon", "Node_ID", "Grid_OD_demand"]]
    grid = grid.sort_values("grid_id").reset_index(drop=True)

    print("========== matched income grid ==========")
    print(f"matched OD-positive nodes: {sum(len(x) for x in grid['Node_ID'])}")
    print(f"total Grid_OD_demand: {grid['Grid_OD_demand'].sum():.6f}")
    print("demand by grid:")
    print(grid[["grid_id", "income", "Grid_OD_demand"]])
    print("=========================================")

    return grid


def generate_matched_income_grid(folder):
    grid = read_income_grid(folder)
    grid = match_nodes_to_income_grid(folder, grid=grid)
    return grid


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def allocate_with_capacity(total_to_allocate, weights, capacity, eps=1e-12):
    """
    Allocate total_to_allocate according to weights, without exceeding the capacity at any position.

    This ensures that:
        sum(e_Grid_OD_demand) == Total_OD_demand * e_rate
        0 <= e_Grid_OD_demand <= Grid_OD_demand
    """
    weights = np.asarray(weights, dtype=float)
    capacity = np.asarray(capacity, dtype=float)

    if total_to_allocate < -eps:
        raise ValueError("total_to_allocate 不能为负。")

    if total_to_allocate > capacity.sum() + eps:
        raise ValueError(
            f"要分配的 EV demand={total_to_allocate} 大于总 capacity={capacity.sum()}。"
        )

    allocation = np.zeros_like(weights, dtype=float)
    remaining = float(total_to_allocate)
    active = capacity > eps

    while remaining > eps and active.any():
        active_weights = weights.copy()
        active_weights[~active] = 0.0

        if active_weights.sum() <= eps:
            active_weights = active.astype(float)

        proposal = remaining * active_weights / active_weights.sum()

        room = capacity - allocation
        add = np.minimum(proposal, room)
        add[~active] = 0.0

        allocation += add
        remaining -= add.sum()

        newly_full = room - add <= eps
        active = active & (~newly_full)

    return allocation


def compute_ev_rate_by_grid_sigmoid(
    e_rate,
    grid,
    beta=2.5,
    open_ended_multiplier=1.25,
    min_score=0.05,
):
    """
    Allocate an EV share to each grid based on income.

    Input:
        e_rate:
            citywide EV OD share; for example, 0.1 means that 10% of total citywide OD demand is assigned to EVs.
        grid:
            The grid returned by match_nodes_to_income_grid().

    Output:
        e_rate_list:
            Values correspond one-to-one with the row order of grid.
    """
    if e_rate < 0 or e_rate > 1:
        raise ValueError("e_rate 必须在 [0, 1] 之间。")

    grid = grid.copy()

    demand = pd.to_numeric(grid["Grid_OD_demand"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total_demand = demand.sum()

    if total_demand <= 0:
        raise ValueError("Grid_OD_demand 总和为 0，无法分配 EV OD。")

    city_midincome = float(pd.to_numeric(grid["midincome"], errors="coerce").dropna().iloc[0])

    rep_income = []

    for income_range in grid["income"]:
        a, b = parse_income_value(income_range)
        a = float(a)

        if b is None or pd.isna(b):
            rep_income.append(a * open_ended_multiplier)
        else:
            rep_income.append(0.5 * (a + float(b)))

    rep_income = np.asarray(rep_income, dtype=float)

    # Log ratio of each income level relative to the citywide median income.
    income_log_ratio = np.log((rep_income + 1.0) / (city_midincome + 1.0))

    # Higher-income grids receive higher scores, while lower-income grids receive lower scores.
    score = sigmoid(beta * income_log_ratio)

    # Ensure that lower-income grids are not assigned zero EV demand.
    score = np.maximum(score, min_score)

    # EV demand allocation weights are determined jointly by OD demand and income scores.
    weights = demand * score

    total_ev_demand = total_demand * float(e_rate)

    e_grid_demand = allocate_with_capacity(
        total_to_allocate=total_ev_demand,
        weights=weights,
        capacity=demand,
    )

    e_rate_list = np.zeros_like(demand, dtype=float)
    nonzero = demand > 0
    e_rate_list[nonzero] = e_grid_demand[nonzero] / demand[nonzero]

    debug = pd.DataFrame({
        "grid_id": grid["grid_id"].to_numpy(),
        "income": grid["income"].to_numpy(),
        "rep_income": rep_income,
        "Grid_OD_demand": demand,
        "income_score": score,
        "e_Grid_OD_demand": e_grid_demand,
        "e_rate": e_rate_list,
    })

    print("========== EV allocation debug ==========")
    print(f"target city e_rate: {e_rate:.6f}")
    print(f"city_midincome: {city_midincome:.6f}")
    print(f"total_demand: {total_demand:.6f}")
    print(f"target total_ev_demand: {total_ev_demand:.6f}")
    print(f"actual total_ev_demand: {e_grid_demand.sum():.6f}")
    print(debug)
    print("=========================================")

    return e_rate_list

def compute_ev_rate_by_grid_linear(
    e_rate,
    grid,
    k=0.15,
    b=0.10,
    open_ended_multiplier=3.0,
    min_score=0.0,
    max_score=1.0,
):
    """
    Allocate an EV share to each grid based on income.

        1. Total citywide EV OD demand = Total_OD_demand * e_rate
        2. Higher-income grids receive a higher allocation preference.
        3. EV OD demand for each grid does not exceed its Grid_OD_demand.
        4. Return e_rate_list in the same order as the rows in grid.
    where:
        x = rep_income / city_midincome - 1
    """

    if e_rate < 0 or e_rate > 1:
        raise ValueError("e_rate 必须在 [0, 1] 之间。")

    grid = grid.copy()

    demand = pd.to_numeric(
        grid["Grid_OD_demand"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    total_demand = demand.sum()

    if total_demand <= 0:
        raise ValueError("Grid_OD_demand 总和为 0，无法分配 EV OD。")

    city_midincome = float(
        pd.to_numeric(grid["midincome"], errors="coerce")
        .dropna()
        .iloc[0]
    )

    if city_midincome <= 0:
        raise ValueError(f"city_midincome 非法: {city_midincome}")

    rep_income = []

    for income_range in grid["income"]:
        a, upper = parse_income_value(income_range)
        a = float(a)

        if upper is None or pd.isna(upper):
            rep_income.append(a * open_ended_multiplier)
        else:
            rep_income.append(0.5 * (a + float(upper)))

    rep_income = np.asarray(rep_income, dtype=float)

    # --------------------------------------------------------
    # Linear score calculation.
    # x is the relative deviation from the citywide median income.
    # --------------------------------------------------------
    x = rep_income / city_midincome

    score_raw = k * x + b

    # Prevent weights for lower-income grids from becoming zero or negative.
    score = np.maximum(score_raw, min_score)

    if max_score is not None:
        score = np.minimum(score, max_score)

    # --------------------------------------------------------
    # EV demand allocation weights are determined jointly by OD demand and income scores.
    # --------------------------------------------------------
    weights = demand * score

    total_ev_demand = total_demand * float(e_rate)

    e_grid_demand = allocate_with_capacity(
        total_to_allocate=total_ev_demand,
        weights=weights,
        capacity=demand,
    )

    e_rate_list = np.zeros_like(demand, dtype=float)
    nonzero = demand > 0
    e_rate_list[nonzero] = e_grid_demand[nonzero] / demand[nonzero]

    actual_city_e_rate = e_grid_demand.sum() / total_demand

    debug = pd.DataFrame({
        "grid_id": grid["grid_id"].to_numpy(),
        "income": grid["income"].to_numpy(),
        "rep_income": rep_income,
        "city_midincome": city_midincome,
        "x": x,
        "score_raw": score_raw,
        "score_used": score,
        "Grid_OD_demand": demand,
        "e_Grid_OD_demand": e_grid_demand,
        "e_rate": e_rate_list,
    })

    print("========== linear-score EV allocation debug ==========")
    print(f"target city e_rate       = {e_rate:.8f}")
    print(f"actual city e_rate       = {actual_city_e_rate:.8f}")
    print(f"k                        = {k:.6f}")
    print(f"b                        = {b:.6f}")
    print(f"city_midincome           = {city_midincome:.6f}")
    print(f"total_demand             = {total_demand:.6f}")
    print(f"target total_ev_demand   = {total_ev_demand:.6f}")
    print(f"actual total_ev_demand   = {e_grid_demand.sum():.6f}")
    print(debug)
    print("======================================================")

    return e_rate_list


def split_od_by_grid_ev_rate(e_rate_list, grid, folder):
    """
    Split OD_Number according to each grid's e_rate into:
        e_OD_Number
        g_OD_Number

    Note:
        The split is based only on the grid associated with O_ID.
    """
    folder = Path(folder)
    od = read_od_data(folder)

    if len(e_rate_list) != len(grid):
        raise ValueError(
            f"e_rate_list 长度 {len(e_rate_list)} 和 grid 长度 {len(grid)} 不一致。"
        )

    node_to_e_rate = {}

    for row_idx, row in grid.reset_index(drop=True).iterrows():
        rate = float(e_rate_list[row_idx])
        node_ids = row["Node_ID"]

        if isinstance(node_ids, str):
            node_ids = json.loads(node_ids)

        for node_id in node_ids:
            node_to_e_rate[int(node_id)] = rate

    od["e_rate"] = od["O_ID"].map(node_to_e_rate).fillna(0.0)

    od["e_OD_Number"] = od["OD_Number"] * od["e_rate"]
    od["g_OD_Number"] = od["OD_Number"] * (1.0 - od["e_rate"])

    return od


def build_od_matrix_from_column(od_data, value_col):
    """
    Find the global minimum and maximum across the O_ID and D_ID columns,
    and generate a matrix with consecutive indices.

    Rows: D_ID
    Columns: O_ID
    Values: value_col
    """
    if value_col not in od_data.columns:
        raise ValueError(f"od_data lack: {value_col}")

    min_id = int(min(od_data["O_ID"].min(), od_data["D_ID"].min()))
    max_id = int(max(od_data["O_ID"].max(), od_data["D_ID"].max()))

    ids = np.arange(min_id, max_id + 1, dtype=int)

    mat = pd.DataFrame(
        0.0,
        index=ids,
        columns=ids,
    )

    grouped = (
        od_data.groupby(["D_ID", "O_ID"], as_index=False)[value_col]
        .sum()
    )

    for row in grouped.itertuples(index=False):
        d_id = int(row.D_ID)
        o_id = int(row.O_ID)
        val = float(getattr(row, value_col))
        mat.at[d_id, o_id] = val

    mat.index.name = "D_ID"
    mat.columns.name = "O_ID"

    return mat


def build_gas_ev_od_matrices(e_rate, grid, folder, func="sigmoid", args=[2.5], save=True):
    """
    Main function:

    1. Allocate e_rate_list based on grid income.
    2. Split OD demand into g_OD_Number and e_OD_Number.
    3. Generate g_od_df and e_od_df.
    4. Save to:
        ./folder/{city_name}_od_matrix_g.csv
        ./folder/{city_name}_od_matrix_e.csv
    """
    folder = Path(folder)
    city_name = get_city_name(folder)
    if func == 'sigmoid':
        beta = args[0]
        e_rate_list = compute_ev_rate_by_grid_sigmoid(e_rate=e_rate, beta=beta, grid=grid)
    elif func == 'linear':
        k = args[0]
        b = args[1]
        e_rate_list = compute_ev_rate_by_grid_linear(e_rate=e_rate, k=k, b=b, grid=grid)


    od_data = split_od_by_grid_ev_rate(
        e_rate_list=e_rate_list,
        grid=grid,
        folder=folder,
    )

    g_od_df = build_od_matrix_from_column(od_data, "g_OD_Number")
    e_od_df = build_od_matrix_from_column(od_data, "e_OD_Number")

    if save:
        # Define the output directory.
        rate = int(e_rate * 20)
        target_dir = folder / "od_matrix" /f"{rate}"


        # parents=True creates any missing parent directories.
        # exist_ok=True prevents an error if the directory already exists.
        target_dir.mkdir(parents=True, exist_ok=True)


        g_path = target_dir / f"{city_name}_od_matrix_g.csv"
        e_path = target_dir / f"{city_name}_od_matrix_e.csv"
        split_path = target_dir / f"{city_name}_od_split_by_ev_rate.csv"

        g_od_df.to_csv(g_path)
        e_od_df.to_csv(e_path)
        od_data.to_csv(split_path, index=False, encoding="utf-8-sig")

        print(f"✅ saved gas OD matrix: {g_path}")
        print(f"✅ saved EV  OD matrix: {e_path}")
        print(f"✅ saved split OD data : {split_path}")

    return g_od_df, e_od_df, od_data, e_rate_list

for city_id in [0, 2, 3, 5, 8, 9, 12, 13, 18, 19]:
    folder = folders[city_id]
    grid = generate_matched_income_grid(folder)
    for i in range(0, 21):
        # g_od_df, e_od_df, od_data, e_rate_list = build_gas_ev_od_matrices(
        #     e_rate=i/20,
        #     grid=grid,
        #     folder=folder,
        #     func="sigmoid",
        #     args=[2.5],
        #     save=True,
        # )

        g_od_df, e_od_df, od_data, e_rate_list = build_gas_ev_od_matrices(
            e_rate=i/20,
            grid=grid,
            folder=folder,
            func="linear",
            args=[0.0, 1.0],   #[k, b] y = kx + b
            save=True,
        )
        print("Total original OD demand:", od_data["OD_Number"].sum())
        print("Total gasoline OD demand:", od_data["g_OD_Number"].sum())
        print("Total EV OD demand:", od_data["e_OD_Number"].sum())
        print("EV share:", od_data["e_OD_Number"].sum() / od_data["OD_Number"].sum())