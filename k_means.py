import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _safe_divide(a, b, eps=1e-12):
    if abs(b) < eps:
        return np.nan
    return a / b


def prepare_data_trend_features(path="./recordtime_gamma_e=1.0.csv"):
    df = pd.read_csv(path)

    first_col = df.columns[0]
    df = df.set_index(first_col)
    df.index = df.index.astype(int)

    required_i = [0, 6, 14, 20]
    for i in required_i:
        if i not in df.index:
            raise ValueError(f"缺少 i={i} 这一行。")

    city_list = {}

    for city in df.columns:
        s = pd.to_numeric(df[city], errors="coerce").sort_index()

        if s.isna().any():
            raise ValueError(f"{city} 存在 NaN。")

        values = s.values.astype(float)

        S0 = float(s.loc[0])
        S03 = float(s.loc[6])
        S07 = float(s.loc[14])
        S1 = float(s.loc[20])

        total = S0 - S1

        total_drop = (S0 - S1) / S0
        early_drop = (S0 - S03) / S0
        mid_drop = (S03 - S07) / S0
        late_drop = (S07 - S1) / S0

        if abs(total) < 1e-12:
            early_share = 0.0
            late_share = 0.0
        else:
            early_share = (S0 - S03) / total
            late_share = (S07 - S1) / total

        monotonic_violation = np.maximum(np.diff(values), 0.0).sum() / S0

        city_list[city] = [
            total_drop,
            early_drop,
            mid_drop,
            late_drop,
            early_share,
            late_share,
            monotonic_violation,
        ]

    return city_list


import re
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize

def _clean_city_name(city):
    """
    Convert 01_San_Francisco to San Francisco.
    """
    city = re.sub(r"^\d+_", "", str(city))
    city = city.replace("_", " ")
    return city

def k_means(
    city_list,
    k=3,
    feature_indices=None,
    feature_weights=None,
    random_state=42,
    save_path="./kmeans_city_clusters.png",
    show=True,
):
    """
    Perform KMeans clustering on city_list and plot the results.

    Parameters:
        city_list:
            Dictionary returned by prepare_data().
            Each city is associated with four metrics.

        k:
            Number of clusters.

        feature_indices:
            Specify which features to use for clustering.
            Examples:
                feature_indices=None       Use all features.
                feature_indices=[0, 1, 2]  Use features 0, 1, and 2.
                feature_indices=[0, 1, 3]  Use features 0, 1, and 3.
                feature_indices=[3]        Use only feature 3.

        feature_weights:
            Apply weights to the selected features.
            The length must match feature_indices.
            Examples:
                feature_indices=[0, 1, 3]
                feature_weights=[2.0, 1.0, 1.5]

    return:
        result_df:
            Metrics and cluster labels for each city.
    """
    if not isinstance(city_list, dict) or len(city_list) == 0:
        raise ValueError("city_list 不能为空。")

    cities = list(city_list.keys())

    X = np.array([city_list[c] for c in cities], dtype=float)

    n_features = X.shape[1]

    if feature_indices is None:
        feature_indices = list(range(n_features))
    else:
        feature_indices = list(feature_indices)

    if len(feature_indices) == 0:
        raise ValueError("feature_indices 不能为空。")

    for idx in feature_indices:
        if idx < 0 or idx >= n_features:
            raise ValueError(
                f"feature_indices 中存在非法索引 {idx}，当前特征数量为 {n_features}。"
            )

    X_used = X[:, feature_indices]

    if np.isnan(X_used).any():
        bad_rows = np.where(np.isnan(X_used).any(axis=1))[0]
        bad_cities = [cities[i] for i in bad_rows]
        raise ValueError(f"以下城市存在 NaN 特征，无法聚类: {bad_cities}")

    if k <= 0:
        raise ValueError("k 必须大于 0。")

    if k > len(cities):
        raise ValueError(f"k={k} 不能大于城市数量 {len(cities)}。")

    # --------------------------------------------------------
    # Standardization.
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_used)

    # --------------------------------------------------------
    # Apply weights after standardization; otherwise, StandardScaler would offset the weights.
    # --------------------------------------------------------
    if feature_weights is not None:
        feature_weights = np.asarray(feature_weights, dtype=float)

        if len(feature_weights) != len(feature_indices):
            raise ValueError(
                f"feature_weights 长度 {len(feature_weights)} "
                f"和 feature_indices 长度 {len(feature_indices)} 不一致。"
            )

        X_scaled = X_scaled * feature_weights

    # --------------------------------------------------------
    # KMeans
    # --------------------------------------------------------
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=random_state,
    )

    labels = model.fit_predict(X_scaled)

    # ========================================================
    # Map cluster labels to semantic types here.
    # ========================================================
    cluster_to_type = {
        0: "Type I",
        1: "Type II",
        2: "Type III",
    }

    # ========================================================
    # Map types to colors.

    # ========================================================
    type_to_color = {
        "Type I": "#F2B701",  # warm yellow
        "Type II": "#3BAA75",  # soft green
        "Type III": "#3B82F6",  # soft blue
    }

    # ========================================================
    # Font and figure settings.
    # ========================================================
    font_family = "Times New Roman"

    point_size = 220  # City marker size
    legend_marker_size = 11  # Legend marker size
    legend_fontsize = 18  # Legend font size
    axis_label_size = 18  # Axis label font size
    tick_label_size = 18  # Tick label font size
    city_label_size = 18  # City label font size

    plt.rcParams["font.family"] = font_family
    plt.rcParams["mathtext.fontset"] = "stix"

    # --------------------------------------------------------
    # Results table.
    # --------------------------------------------------------
    result_df = pd.DataFrame(
        X,
        index=cities,
    )

    result_df["cluster"] = labels
    result_df["type"] = [cluster_to_type[int(label)] for label in labels]
    result_df["used_features"] = ",".join([str(i) for i in feature_indices])
    result_df = result_df.sort_values(["type", "cluster"])

    print("========== KMeans clustering result ==========")
    print(f"used feature indices: {feature_indices}")
    print("cluster_to_type:")
    print(cluster_to_type)
    print(result_df)
    print("==============================================")

    # --------------------------------------------------------
    # Plotting.
    # If more than two features are used, apply PCA to reduce them to two dimensions for visualization.
    # --------------------------------------------------------
    used_dim = len(feature_indices)

    if used_dim == 1:
        plot_x = X_scaled[:, 0]
        plot_y = np.zeros_like(plot_x)
        xlabel = f"Feature {feature_indices[0]} standardized"
        ylabel = ""

    elif used_dim == 2:
        plot_x = X_scaled[:, 0]
        plot_y = X_scaled[:, 1]
        xlabel = f"Feature {feature_indices[0]} standardized"
        ylabel = f"Feature {feature_indices[1]} standardized"

    else:
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        plot_x = X_pca[:, 0]
        plot_y = X_pca[:, 1]
        xlabel = rf"$PC_1$ ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
        ylabel = rf"$PC_2$ ({pca.explained_variance_ratio_[1] * 100:.1f}%)"

    # --------------------------------------------------------
    # Create the plot.
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    city_types = [cluster_to_type[int(label)] for label in labels]

    type_to_code = {
        "Type I": 0,
        "Type II": 1,
        "Type III": 2,
    }

    type_codes = np.array([type_to_code[t] for t in city_types])

    cmap = LinearSegmentedColormap.from_list(
        "type_gradient",
        [
            type_to_color["Type I"],
            type_to_color["Type II"],
            type_to_color["Type III"],
        ],
        N=256,
    )

    norm = Normalize(vmin=0, vmax=2)

    scatter = ax.scatter(
        plot_x,
        plot_y,
        c=type_codes,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=0.92,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )

    # --------------------------------------------------------
    # City labels are initially placed next to their markers.
    # --------------------------------------------------------
    texts = []

    for city, x, y in zip(cities, plot_x, plot_y):
        city_name = _clean_city_name(city)

        text = ax.text(
            x + 0.04,  # Initial horizontal offset.
            y + 0.04,  # Initial vertical offset.
            city_name,
            fontsize=city_label_size,
            fontfamily=font_family,
            ha="left",
            va="bottom",
            zorder=5,
            clip_on=True,
        )
        texts.append(text)

    # --------------------------------------------------------
    # Prevent city labels from overlapping and use arrows to point to their markers.
    # --------------------------------------------------------
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            x=plot_x,
            y=plot_y,
            ax=ax,

            # Repulsion between text labels.
            force_text=(1.2, 1.5),

            # Repulsion between text labels and data points.
            force_static=(0.8, 1.0),

            # Gently pull labels back toward their corresponding points.
            force_pull=(0.02, 0.02),

            # Reduce the initial separation force.
            force_explode=(1.0, 1.2),

            # Use a moderate expansion range.
            expand=(1.2, 1.6),

            ensure_inside_axes=True,
            expand_axes=True,

            only_move={
                "text": "xy",
                "static": "xy",
                "explode": "xy",
                "pull": "xy",
            },

            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                lw=0.6,
                alpha=0.65,
                shrinkA=3,
                shrinkB=4,
            ),

            # iter_lim=1000,
        )

    except ImportError:
        print("⚠️ adjustText is not installed, so city labels may overlap. Install it with: pip install adjustText")

    # --------------------------------------------------------
    # Axis settings.
    # --------------------------------------------------------
    ax.set_xlabel(xlabel, fontsize=axis_label_size, fontfamily=font_family)
    ax.set_ylabel(ylabel, fontsize=axis_label_size, fontfamily=font_family)

    # No plot title.
    # ax.set_title(...)

    # Disable background grid lines.
    ax.grid(False)

    # Direct axis ticks inward.
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        labelsize=tick_label_size,
    )

    # Use Times New Roman for axis tick labels.
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(font_family)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily(font_family)

    # Add margins to prevent labels from touching the edges or being clipped.
    ax.margins(x=0.25, y=0.25)

    # --------------------------------------------------------


    cbar = plt.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label("")
    cbar.ax.set_title("")

    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Type I", "Type II", "Type III"])

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily(font_family)
        tick.set_fontsize(tick_label_size)

    # --------------------------------------------------------
    # Legend: retain only Type I, Type II, and Type III.
    # --------------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Type I",
            markerfacecolor=type_to_color["Type I"],
            markeredgecolor="none",
            markersize=legend_marker_size,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Type II",
            markerfacecolor=type_to_color["Type II"],
            markeredgecolor="none",
            markersize=legend_marker_size,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Type III",
            markerfacecolor=type_to_color["Type III"],
            markeredgecolor="none",
            markersize=legend_marker_size,
        ),
    ]

    # ax.legend(
    #     handles=legend_handles,
    #     fontsize=legend_fontsize,
    #     frameon=False,
    #     loc="best",
    #     prop={"family": font_family, "size": legend_fontsize},
    #
    #     handletextpad=0.25,  # Spacing between the marker and the Type X label.
    #     handlelength=0.8,  # Horizontal length allocated to the legend marker.
    #     borderpad=0.2,  # Legend padding.
    #     labelspacing=0.35,  # Vertical spacing between legend entries.
    # )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ already save: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return result_df


if __name__ == "__main__":
    path = "./recordtime_gamma_e=1.0.csv"

    city_list = prepare_data_trend_features("./recordtime_gamma_e=1.0.csv")

    result_df = k_means(
        city_list,
        k=3,
        feature_indices=[0, 1, 2, 3, 4, 5, 6],
        feature_weights=[1.0, 0.8, 1.0, 1.2, 0.6, 1.2, 2.0],
        save_path="./kmeans_city_trend_clusters.png",
        show=True,
    )

    result_df.to_csv("./kmeans_city_clusters_result.csv", encoding="utf-8-sig")
    print("✅ already save: ./kmeans_city_clusters_result.csv")