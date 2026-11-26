from pathlib import Path
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


@dataclass
class WResult:
    W: csr_matrix
    link_ids: np.ndarray


def cal_w(path1=None, city=None):
    """
    Build the link-link adjacency matrix W from network.csv.

    Two links are considered adjacent if:
        they share at least one endpoint node (i.e., their a_node or b_node values intersect).

    Row and column order of W:
        link_id in ascending order.

    return:
        WResult:
            W: scipy sparse csr_matrix, shape=[num_links, num_links]
            link_ids: np.ndarray, the link_id order corresponding to the rows and columns of W
    """
    if path1 is None:
        if city is None:
            raise ValueError("path1 和 city 至少需要提供一个。")
        path1 = Path(city) / "03_AequilibraE_results" / "network.csv"
    else:
        path1 = Path(path1)

    if not path1.exists():
        raise FileNotFoundError(f"找不到 network 文件: {path1}")

    net = pd.read_csv(path1)

    required_cols = ["a_node", "b_node", "link_id"]
    missing = [c for c in required_cols if c not in net.columns]
    if missing:
        raise ValueError(f"{path1} 缺少字段: {missing}")

    net = net[required_cols].copy()
    net["a_node"] = net["a_node"].astype(int)
    net["b_node"] = net["b_node"].astype(int)
    net["link_id"] = net["link_id"].astype(int)

    net = net.sort_values("link_id").reset_index(drop=True)

    link_ids = net["link_id"].to_numpy(dtype=int)
    n = len(link_ids)

    # link_id -> row index
    link_id_to_idx = {lid: idx for idx, lid in enumerate(link_ids)}

    # node -> incident link row indices
    node_to_links = {}

    for row in net.itertuples(index=False):
        a = int(row.a_node)
        b = int(row.b_node)
        lid = int(row.link_id)
        idx = link_id_to_idx[lid]

        node_to_links.setdefault(a, []).append(idx)
        node_to_links.setdefault(b, []).append(idx)

    rows = []
    cols = []

    # For each node, every pair of incident links is considered adjacent.
    for node, incident in node_to_links.items():
        incident = sorted(set(incident))

        if len(incident) <= 1:
            continue

        for i, j in combinations(incident, 2):
            rows.append(i)
            cols.append(j)

            rows.append(j)
            cols.append(i)

    data = np.ones(len(rows), dtype=np.float64)

    W = coo_matrix(
        (data, (rows, cols)),
        shape=(n, n),
        dtype=np.float64,
    ).tocsr()

    # Two links may be counted more than once if they share both a_node and b_node.
    # Moran's I requires binary adjacency, so all nonzero weights are set to 1.
    W.data[:] = 1.0

    # Remove any possible self-loops.
    W.setdiag(0.0)
    W.eliminate_zeros()

    print("========== W constructed ==========")
    print(f"path: {path1}")
    print(f"num_links: {n}")
    print(f"num_nonzero_w: {W.nnz}")
    print(f"sum_w: {W.sum():.0f}")
    print("===================================")

    return WResult(W=W, link_ids=link_ids)


def _find_value_col(df, preferred_col="VOC_max"):
    """
    Automatically identify the VOC field.
    Use preferred_col first.
    If it is unavailable, try common field names.
    """
    candidates = [
        preferred_col,
        "VOC_max",
        "VOCmax",
        "VOC_MAX",
        "VOC_AB",
        "VOC",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"找不到 VOC 字段。当前字段有: {list(df.columns)}"
    )


def _read_assignment_as_series(path, link_ids, value_col="VOC_max"):
    """
    Read assignment_result_x.csv and
    return the VOC series ordered according to link_ids.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 assignment result 文件: {path}")

    df = pd.read_csv(path)

    if "link_id" not in df.columns:
        raise ValueError(f"{path} 缺少 link_id 字段。")

    col = _find_value_col(df, preferred_col=value_col)

    df = df[["link_id", col]].copy()
    df["link_id"] = df["link_id"].astype(int)
    df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[col].isna().any():
        raise ValueError(f"{path} 的 {col} 存在 NaN。")

    # Aggregate rows first if a link_id appears more than once.
    s = df.groupby("link_id")[col].sum()

    missing = [int(lid) for lid in link_ids if int(lid) not in s.index]
    if missing:
        print("The assignment result is missing the following link_id values:")
        print(missing[:200])
        if len(missing) > 200:
            print(f"... and {len(missing) - 200} more not shown")
        raise ValueError(f"{path} 缺少 {len(missing)} 个 link_id。")

    s = s.reindex(link_ids)

    return s.to_numpy(dtype=float)


def global_moran(w_result, path2, path3, value_col="VOC_max"):
    """
    Calculate Moran's I.

    path2:
        Assignment results when EV penetration = 0.
    path3:
        Assignment results when EV penetration = 1.

    x = VOC_1 - VOC_0

    Moran_I =
        (n / sum_ij w_ij)
        *
        (sum_i sum_j w_ij (x_i - x_bar)(x_j - x_bar))
        /
        sum_i (x_i - x_bar)^2
    """
    W = w_result.W
    link_ids = w_result.link_ids

    list_0 = _read_assignment_as_series(
        path=path2,
        link_ids=link_ids,
        value_col=value_col,
    )

    list_1 = _read_assignment_as_series(
        path=path3,
        link_ids=link_ids,
        value_col=value_col,
    )

    x = abs((list_1 - list_0) / (list_0 + 1e-12))
    # x = abs(list_1 - list_0)

    n = len(x)
    x_bar = x.mean()
    z = x - x_bar

    sum_w = float(W.sum())
    denominator = float(np.dot(z, z))

    if sum_w <= 0:
        raise ValueError("W 的权重总和为 0，无法计算 Moran's I。")

    if denominator <= 1e-12:
        raise ValueError("x 的方差接近 0，无法计算 Moran's I。")

    numerator = float(z @ (W @ z))

    moran_i = (n / sum_w) * (numerator / denominator)

    return moran_i


def _resolve_assignment_path(city, folder_i, file_i):
    """
    First look for assignment_result_{file_i}.csv.
    If it does not exist, use the only assignment_result_*.csv file in folder_i.
    """
    city = Path(city)
    folder = city / "04_my_AequilibraE_results" / str(folder_i)

    target = folder / f"assignment_result_{file_i}.csv"

    if target.exists():
        return target

    candidates = sorted(folder.glob("assignment_result_*.csv"))

    if len(candidates) == 1:
        print(f"Could not find {target}; using {candidates[0]} instead.")
        return candidates[0]

    raise FileNotFoundError(
        f"找不到 {target}，且 {folder} 下 assignment_result_*.csv 数量不是 1。"
    )


def calculate_moran_for_cities(folders, city_ids, value_col="VOC_max"):
    """
    Calculate Moran's I for the specified city_ids.

    path1:
        ./{city}/03_AequilibraE_results/network.csv

    path2:
        ./{city}/04_my_AequilibraE_results/0/assignment_result_0.csv
        or the actual filename in use.

    path3:
        ./{city}/04_my_AequilibraE_results/20/assignment_result_20.csv
    """
    moran_dict = {}

    for idx in city_ids:
        city = folders[idx]

        print(f"\n========== calculating {city} ==========")

        path1 = Path(city) / "03_AequilibraE_results" / "network.csv"

        # Note:


        path2 = _resolve_assignment_path(city, folder_i=0, file_i=0)
        path3 = _resolve_assignment_path(city, folder_i=20, file_i=20)

        w_result = cal_w(path1=path1, city=city)

        moran_i = global_moran(
            w_result=w_result,
            path2=path2,
            path3=path3,
            value_col=value_col,
        )

        moran_dict[city] = moran_i

        print(f"{city} Moran's I = {moran_i:.8f}")

    return moran_dict


if __name__ == "__main__":
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

    city_ids = [0, 2, 3, 5, 8, 9, 12, 13, 18, 19]

    moran_dict = calculate_moran_for_cities(
        folders=folders,
        city_ids=city_ids,
        value_col="VOC_max",
    )

    print("\n========== Final Moran's I ==========")
    print(moran_dict)

    pd.DataFrame(
        [{"city": city, "Moran_I": val} for city, val in moran_dict.items()]
    ).to_csv("moran_i_results.csv", index=False, encoding="utf-8-sig")

    print("✅ saved: moran_i_results.csv")