import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import geopandas as gpd
import numpy as np

from copy import deepcopy as deep
from scipy.io import savemat
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths import TrafficClass
from concurrent.futures import ProcessPoolExecutor, as_completed


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





# Create a 22x20 matrix using NumPy
record_matrix = np.zeros((23, 20))

# Access elements of the matrix
## Original
alpha_precondition = [0.5,0.6,0.5,0.5,0.5,0.6,0.5,0.15,0.6,0.5,0.25,0.5,0.25,0.5,0.5,0.5,0.2,0.15,0.5,0.5]
beta_precondition  = [1.8,3.0,1.2,1.3,1.2,1.8,1.5,1.80,1.3,1.5,1.50,1.5,2.00,1.2,2.0,1.5,1.5,1.20,1.5,1.5]

# 1. alpha -20%, beta 0%
# alpha_precondition = [0.4, 0.6, 0.4, 0.4, 0.5, 0.48, 0.5, 0.15, 0.48, 0.4, 0.25, 0.5, 0.2, 0.4, 0.5, 0.5, 0.2, 0.15, 0.4, 0.4]
# beta_precondition = [1.8, 3.0, 1.2, 1.3, 1.2, 1.8, 1.5, 1.80, 1.3, 1.5, 1.50, 1.5, 2.00, 1.2, 2.0, 1.5, 1.5, 1.20, 1.5, 1.5]

# 2. alpha +20%, beta 0%
# alpha_precondition = [0.6, 0.6, 0.6, 0.6, 0.5, 0.72, 0.5, 0.15, 0.72, 0.6, 0.25, 0.5, 0.3, 0.6, 0.5, 0.5, 0.2, 0.15, 0.6, 0.6]
# beta_precondition = [1.8, 3.0, 1.2, 1.3, 1.2, 1.8, 1.5, 1.80, 1.3, 1.5, 1.50, 1.5, 2.00, 1.2, 2.0, 1.5, 1.5, 1.20, 1.5, 1.5]

# 3. alpha 0%, beta -20%
# alpha_precondition = [0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.5, 0.15, 0.6, 0.5, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5, 0.2, 0.15, 0.5, 0.5]
# beta_precondition = [1.44, 3.0, 1.0, 1.04, 1.2, 1.44, 1.5, 1.80, 1.04, 1.20, 1.50, 1.5, 1.60, 1.0, 2.0, 1.5, 1.5, 1.20, 1.20, 1.20]

# 4. alpha 0%, beta +20%
# alpha_precondition = [0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.5, 0.15, 0.6, 0.5, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5, 0.2, 0.15, 0.5, 0.5]
# beta_precondition = [2.16, 3.0, 1.44, 1.56, 1.2, 2.16, 1.5, 1.80, 1.56, 1.80, 1.50, 1.5, 2.40, 1.44, 2.0, 1.5, 1.5, 1.20, 1.80, 1.80]



road_1_capacity_precondition = [2200,2200,2200,2200,2000,2200,2200,2200,2200,2200,2200,1800,2200,2000,2200,1800,2200,2200,2000,2200]
road_1_speed_precondition = [90,90,90,90,90,90,90,90,90,90,90,60,60,90,90,65,70,90,90,90]       # free flow speed (km/h)
road_2_capacity_precondition = [2000,2000,2000,2000,1900,2000,2000,2000,2000,2000,2000,1500,2000,1800,2000,1500,2000,2000,1800,2000]
road_2_speed_precondition = [60,65,65,60,60,60,65,65,65,65,60,40,45,60,60,50,50,65,60,60]
road_3_capacity_precondition = [1400,1400,1400,1400,1400,1400,1400,1300,1400,1400,1400,600,1300,1200,1200,900,1400,1500,1300,1400]
road_3_speed_precondition = [40,45,45,40,40,40,40,40,45,45,40,25,30,30,30,35,35,45,35,40]
od_multiplier_precondition = [0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]

city_params= {
    folder: {
        'alpha': alpha,
        'beta': beta,
        'road_1_capacity': road_1_capacity,
        'road_1_speed': road_1_speed,
        'road_2_capacity': road_2_capacity,
        'road_2_speed': road_2_speed,
        'road_3_capacity': road_3_capacity,
        'road_3_speed': road_3_speed,
        'od_multiplier': od_multiplier
    }
    for folder, alpha, beta, road_1_capacity, road_1_speed, road_2_capacity, road_2_speed, road_3_capacity, road_3_speed, od_multiplier in zip(
        folders,
        alpha_precondition,
        beta_precondition,
        road_1_capacity_precondition,
        road_1_speed_precondition,
        road_2_capacity_precondition,
        road_2_speed_precondition,
        road_3_capacity_precondition,
        road_3_speed_precondition,
        od_multiplier_precondition
    )
}



'''----------------------------------------------AEM calculation-------------------------------------------------------------'''

## Generate the corresponding AEM based on the OD CSV file
# od_csv_g to aem_g
# od_csv_e to aem_e
def generate_aem_from_matrix(city_folder, od_csv, multiplier=0.6, centroids=None):
    # 1. Read the CSV file
    od_df = pd.read_csv(od_csv, index_col=0)
    od_df = od_df.T
    # 2. Special handling for Pittsburgh (exclude missing centroids)
    if city_folder == '03_Pittsburgh':
        missing_centroids = [61, 62, 63, 101, 102]
        missing_ids = [x + 9999999 for x in missing_centroids]
        od_df = od_df.drop(index=missing_ids, errors='ignore')
        od_df = od_df.drop(columns=[str(x) for x in missing_ids], errors='ignore')

    # Get the number of zones
    zones = list((od_df.index.astype(int) - 9999999).astype(int))
    # Create the AEM matrix
    mat = AequilibraeMatrix()
    mat.create_empty(
        zones=len(zones),
        matrix_names=["matrix"],
        memory_only=True
    )
    mat.index[:] = np.array(zones, dtype=np.uint32)
    core_index = mat.names.index("matrix")
    # 4. Activate the computational view and populate it with OD data
    mat.computational_view(["matrix"])
    # Note: The original 0.6 coefficient is retained and multiplied by the externally provided multiplier (i.e., the penetration rate).
    mat.matrix_view[:, :] = od_df.values.astype(np.float32) * multiplier

    # 5. Set the centroids of the network Graph corresponding to the index
    mat.index = centroids

    # 6. Return the object directly (do not call mat.close(), or the data in memory will be cleared)
    return mat




'''----------------------------------------------AEM calculation-------------------------------------------------------------'''
def update_record_time_csv(record_csv_path, city_path, i, average_traffic_time):
    """
    CSV format:
        Row index = i
        Column = city_path
        Value = average_traffic_time

    Each iteration reads the existing CSV, updates one cell, and overwrites the file.
    """
    if os.path.exists(record_csv_path):
        record_df = pd.read_csv(record_csv_path, index_col=0)
        record_df.index = record_df.index.astype(int)
    else:
        record_df = pd.DataFrame(index=range(21))
        record_df.index.name = "i"

    if city_path not in record_df.columns:
        record_df[city_path] = np.nan

    record_df.loc[int(i), city_path] = float(average_traffic_time)

    record_df = record_df.sort_index()
    record_df.to_csv(record_csv_path, encoding="utf-8-sig")

    return record_df


def worker(city_ind, i, gamma_g, gamma_e, save_record):
    ########   step 1:  identify the city and input file path       #############
    city_path = folders[city_ind]
    print(city_path, i)

    AE_path = city_path + '/' + '03_AequilibraE_results'

    save_path = city_path + '/' + '04_my_AequilibraE_results' + '/' + str(i)
    os.makedirs(save_path, exist_ok=True)

    ########   step 2:  define the od matrix file and generate od index      #############
    od_data = pd.read_csv(city_path + '/' + '01_Input_data' + '/' + city_path[3:] + '_od.csv')
    zones = int(max(od_data.O_ID.max() - 9999999, od_data.D_ID.max() - 9999999))
    index = np.arange(zones) + 1

    #######   step 3: define the trafffic assignment paramters    #####
    connector_speed = 5
    road_1_capacity = road_1_capacity_precondition[city_ind]
    road_1_speed = road_1_speed_precondition[city_ind]

    road_2_capacity = road_2_capacity_precondition[city_ind]
    road_2_speed = road_2_speed_precondition[city_ind]

    road_3_capacity = road_3_capacity_precondition[city_ind]
    road_3_speed = road_3_speed_precondition[city_ind]
    alpha = alpha_precondition[city_ind]  # alpha in BPR function
    beta = beta_precondition[city_ind]  # beta in BPR function
    od_multiplier = 0.6

    # Define your multiplier k
    # k_g_list = {
    #     1: 0.553021735618328, 3: 0.5271, 4: 0.5345, 6: 0.5034, 9: 0.504107273789255,
    #     10: 0.5069, 13: 0.5111, 14: 0.512930766649683, 19: 0.5036, 20: 0.5449,
    # }
    #
    # k_e_list = {
    #     1: 0.196351412267254, 3: 0.1846, 4: 0.1903, 6: 0.1819, 9: 0.188051454181669,
    #     10: 0.1937, 13: 0.1955, 14: 0.191706481595504, 19: 0.1893, 20: 0.234,
    # }
    k_g_list = {
        1: 0.5530, 3: 0.5271, 4: 0.5345, 6: 0.5034, 9: 0.5041,
        10: 0.5069, 13: 0.5111, 14: 0.5129, 19: 0.5241, 20: 0.5449,
    }

    k_e_list = {
        1: 0.1964, 3: 0.1846, 4: 0.1903, 6: 0.1819, 9: 0.1881,
        10: 0.1937, 13: 0.1955, 14: 0.1917, 19: 0.1882, 20: 0.234,
    }



    k_g = k_g_list[city_ind + 1]
    k_e = k_e_list[city_ind + 1]

    e_rate = i / 20

    ########   step 4:  select the initial network shapefile path      #############
    net = gpd.read_file(AE_path + '/' + city_path[3:] + '.shp')
    net.drop('geometry', axis=1, inplace=True)
    net.columns = ["ID", "DIR", "length", "LINK_ID", "a_node", "b_node", "capacity", "LENGTH1", "speed", "Lanes",
                   "link_type"]

    ########   step 5:  modify the parameters for traffic assignment    #############
    ########### connector ##############
    net.loc[net['link_type'] > 10, 'speed'] = connector_speed
    ########### road class 1 ##############
    net.loc[net['link_type'].isin([1, 2]), 'capacity'] = road_1_capacity
    net.loc[net['link_type'].isin([1, 2]), 'speed'] = road_1_speed
    ########### road class 2 ##############
    net.loc[net['link_type'] == 3, 'capacity'] = road_2_capacity
    net.loc[net['link_type'] == 3, 'speed'] = road_2_speed
    ########### road class 3 ##############
    net.loc[net['link_type'].isin([4, 5]), 'capacity'] = road_3_capacity
    net.loc[net['link_type'].isin([4, 5]), 'speed'] = road_3_speed

    ########### BPR function ##############
    net['free_flow_time'] = net['length'] / net['speed'] * 60
    net['a'] = alpha
    net['b'] = beta

    # Calculate the fixed cost for each link and add it as a new column
    net['fixed_cost_gas'] = net['length'] * k_g
    net['fixed_cost_ele'] = net['length'] * k_e

    ########### organize the network data fields ##############
    network = net[['a_node', 'b_node', "capacity", 'free_flow_time', "a", "b", "fixed_cost_gas", "fixed_cost_ele"]]
    network = network.assign(direction=1)
    network["link_id"] = network.index + 1
    network = network.astype({"a_node": "int64", "b_node": "int64"})

    ########### step 6: conduct the traffic assignment in AE ##############
    ## Build the gasoline-vehicle network graph
    g = Graph()
    g.cost = network['free_flow_time'].values
    g.capacity = network['capacity'].values
    g.free_flow_time = network['free_flow_time'].values

    g.network = network
    g.network_ok = True
    g.status = 'OK'
    g.prepare_graph(index)
    g.set_graph("free_flow_time")
    g.cost = np.array(g.cost, copy=True)
    g.set_skimming(["free_flow_time"])
    g.set_blocked_centroid_flows(False)
    g.network["id"] = g.network.link_id
    g.graph.loc[:, 'fixed_cost_gas'] = net['fixed_cost_gas'].values
    g.graph.loc[:, 'fixed_cost_ele'] = net['fixed_cost_ele'].values

    ## Build the electric-vehicle network graph
    g1 = Graph()
    g1.cost = network['free_flow_time'].values
    g1.capacity = network['capacity'].values
    g1.free_flow_time = network['free_flow_time'].values

    g1.network = network
    g1.network_ok = True
    g1.status = 'OK'
    g1.prepare_graph(index)
    g1.set_graph("free_flow_time")
    g1.cost = np.array(g.cost, copy=True)
    g1.set_skimming(["free_flow_time"])
    g1.set_blocked_centroid_flows(False)
    g1.network["id"] = g.network.link_id
    g1.graph.loc[:, 'fixed_cost_ele'] = net['fixed_cost_ele'].values

    ## Load and calculate the AEM
    od_matrix_csv_g = os.path.join(city_path, "od_matrix", str(i), f"{city_path[3:]}_od_matrix_g.csv")
    od_matrix_csv_e = os.path.join(city_path, "od_matrix", str(i), f"{city_path[3:]}_od_matrix_e.csv")
    aem_g = generate_aem_from_matrix(city_path, od_matrix_csv_g, centroids=g.centroids)
    aem_e = generate_aem_from_matrix(city_path, od_matrix_csv_e, centroids=g1.centroids)

    assigclass_g = TrafficClass("gas_car", g, aem_g)
    assigclass_g.set_fixed_cost(field_name="fixed_cost_gas")
    assigclass_g.set_vot(gamma_g)
    assigclass_e = TrafficClass("ele_car", g1, aem_e)
    assigclass_e.set_fixed_cost(field_name="fixed_cost_ele")
    assigclass_e.set_vot(gamma_e)

    assig = TrafficAssignment()
    assig.set_classes([assigclass_g, assigclass_e])
    assig.set_vdf("bpr")
    assig.set_vdf_parameters({"alpha": "a", "beta": "b"})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm("bfw")
    assig.max_iter = 500
    assig.rgap_target = 0.001
    assig.execute()

    assignment_results = assig.results()
    network_data = network.copy()
    assignment_results.reset_index(inplace=True)
    assignment_results.rename(columns={"index": "link_id"}, inplace=True)

    network_data = network_data.merge(assignment_results[["link_id", "VOC_AB", "PCE_tot"]], on="link_id")

    a = alpha_precondition[city_ind]
    b = beta_precondition[city_ind]

    def calculate_congested_time(row):
        if row["a_node"] > 100000 or row["b_node"] > 100000:
            return row["free_flow_time"]
        else:
            return row["free_flow_time"] * (1 + a * (row["VOC_AB"]) ** b)

    network_data["congested_time"] = network_data.apply(calculate_congested_time, axis=1)

    od_matrix_g = aem_g.matrix["matrix"]
    od_matrix_e = aem_e.matrix["matrix"]

    total_od_flow_g = od_matrix_g.sum().sum()
    total_od_flow_e = od_matrix_e.sum().sum()
    total_od_flow = total_od_flow_g + total_od_flow_e

    total_travel_time = (network_data["congested_time"] * network_data["PCE_tot"]).sum()
    total_traffic_volume = network_data["PCE_tot"].sum()

    average_traffic_time = total_travel_time / total_od_flow

    print(f"[{city_path} - Rate {i}] Average Traffic Time:", average_traffic_time, "minutes")
    print(f"[{city_path} - Rate {i}] DEBUG: Total travel time = {total_travel_time}")
    print(f"[{city_path} - Rate {i}] DEBUG: Total OD flow = {total_od_flow}")

    if save_record:
        # update_record_time_csv, which could cause conflicts, has been removed.
        # This to_csv call will not cause conflicts because save_path is a separate folder for each i.
        network.to_csv(f'{save_path}\\network_{i}.csv')
        assig.results().to_csv(f'{save_path}\\assignment_result_{i}.csv')

    # [Key change 1]: Instead of writing directly to the global CSV, return the data to be recorded as a dictionary to the main process.
    return {
        "city_ind": city_ind,
        "city_path": city_path,
        "rate_i": i,
        "average_traffic_time": average_traffic_time
    }


# =====================================================================
# Parallel task scheduler: replaces the original nested for loops
# =====================================================================
import multiprocessing
def fun(city_ids=[0], i_range=[0, 20], gamma_g=1.0, gamma_e=1.2, save_record=False, max_workers=6):
    tasks = []
    for city_ind in city_ids:
        for i in i_range:
            tasks.append((city_ind, i, gamma_g, gamma_e, save_record))

    print(f"🚀 Generated {len(tasks)} parallel tasks in total. Using {max_workers} CPU cores for concurrent computation...")

    # [Key change 2]: Initialize a list to collect the results returned by all workers.
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, *task): task for task in tasks}

        for future in as_completed(futures):
            task_info = futures[future]
            try:
                # [Key change 3]: Retrieve the dictionary returned by the worker and append it to the list.
                result_data = future.result()
                all_results.append(result_data)
                print(f"✅ Task [{result_data['city_path']} - Rate {result_data['rate_i']}] completed and results collected!")
            except Exception as exc:
                print(f"❌ Task {task_info} failed with exception: {exc}")

    # [Key change 4]: After all parallel tasks finish, the main process safely writes the collected results to a single CSV file.
    if save_record and all_results:
        print("\n💾 Generating the 2D matrix and saving it to recordtime_parallel.csv ...")

        # 1. Convert the collected results into a DataFrame
        df_records = pd.DataFrame(all_results)

        # 2. 👑 Key fix: Use pivot to reconstruct the 2D table!
        # index (rows) represents the penetration rate i
        # columns represent city names
        # values represent the calculated travel times
        matrix_df = df_records.pivot(index='rate_i', columns='city_path', values='average_traffic_time')

        # 3. Save in the familiar format (preserving row labels, i.e., 0 to 20)
        matrix_df.to_csv(f"recordtime_gamma_e={gamma_e}.csv", index=True)
        print("🎉 All tasks completed! The matrix-formatted recordtime.csv has been generated successfully!")




# 1 3 4 6 9 10 13 14 19 20    --> 0 2 3 5 8 9 12 13 18 19

if __name__ == '__main__':
    fun(city_ids = [0, 2, 3, 5, 8, 9, 12, 13, 18, 19], i_range = list(range(0, 21)), gamma_g = 1.0, gamma_e = 1.0, save_record=True)

    # fun(city_ids=[0, 2, 3, 5, 8, 9, 12, 13, 18, 19], i_range=list(range(0, 21)), gamma_g=1.0, gamma_e=1.30, save_record=True)