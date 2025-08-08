import os                     # For file path operations
import pandas as pd           # For reading and manipulating CSV data
import numpy as np            # For numerical operations and matrices
import random                 # For random selections (ant starting points, next city choice)
import matplotlib.pyplot as plt  # For plotting results (currently not used in this script)

# --------- HELPER FUNCTIONS ---------

def combine_costs(time_matrix, vehicle_matrix, w_time, w_cong):
    """
    Combine time and congestion cost matrices into a single weighted cost matrix.
    w_time  - weight for time cost
    w_cong  - weight for congestion cost
    """
    return w_time * time_matrix + w_cong * vehicle_matrix

def initialize_pheromone(n):
    """
    Initialize the pheromone matrix for n nodes.
    Start with pheromone level = 1 on all edges.
    """
    return np.ones((n, n))

def select_next_city(current_city, unvisited, pheromone, visibility, alpha, beta):
    """
    Select the next city based on pheromone strength and visibility.
    alpha - influence of pheromone
    beta  - influence of visibility (1/cost)
    """
    pher = pheromone[current_city, unvisited] ** alpha   # Pheromone influence
    vis = visibility[current_city, unvisited] ** beta    # Visibility influence
    probs = pher * vis                                   # Combined desirability

    if probs.sum() == 0:                                 # If no path has attractiveness
        return random.choice(unvisited)                  # Pick a random unvisited city

    probs = probs / probs.sum()                          # Normalize probabilities
    return np.random.choice(unvisited, p=probs)          # Select next city by probability

def construct_tour(n, pheromone, visibility, alpha, beta):
    """
    Construct a complete tour for an ant starting from a random city.
    """
    unvisited = list(range(n))                           # All cities unvisited
    current = random.choice(unvisited)                   # Random starting city
    tour = [current]                                     # Record starting point
    unvisited.remove(current)                            # Mark as visited

    while unvisited:                                     # Until all cities visited
        nxt = select_next_city(current, unvisited, pheromone, visibility, alpha, beta)
        tour.append(nxt)                                 # Add city to tour
        unvisited.remove(nxt)                            # Mark visited
        current = nxt                                    # Move to new city
    return tour

def local_pheromone_update(pheromone, tour, rho):
    """
    Locally evaporate pheromone along the path taken by an ant.
    rho - local evaporation rate
    """
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]         # Consecutive cities in tour
        pheromone[a][b] *= (1 - rho)                      # Evaporate pheromone
        pheromone[b][a] = pheromone[a][b]                 # Symmetric update (undirected)

def global_pheromone_update(pheromone, best_tour, best_cost, Q):
    """
    Globally reinforce pheromone along the best tour found so far.
    Q - pheromone deposit constant
    """
    for i in range(len(best_tour)):
        a, b = best_tour[i], best_tour[(i + 1) % len(best_tour)]
        pheromone[a][b] += Q / best_cost                  # Add pheromone based on quality
        pheromone[b][a] = pheromone[a][b]                 # Symmetric update

def tour_cost(tour, cost_matrix):
    """
    Compute the total cost of a tour given a cost matrix.
    """
    total = 0.0
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        total += cost_matrix[a][b]
    return total

def tour_length(tour, matrix):
    """
    Same as tour_cost but returns infinity if any segment is unreachable.
    """
    total = 0.0
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        cost = matrix[a][b]
        if not np.isfinite(cost):                         # If invalid path
            return float('inf')                           # Mark as unreachable
        total += cost
    return total

def dominates(cost1, cost2):
    """
    Check if cost1 Pareto-dominates cost2.
    cost1 dominates cost2 if it is <= in all objectives and < in at least one.
    """
    return all(a <= b for a, b in zip(cost1, cost2)) and any(a < b for a, b in zip(cost1, cost2))

# --------- MAIN MULTI-OBJECTIVE ACO ---------

def ACO_multi_objective(time_matrix, vehicle_matrix, w_time=0.5, w_cong=0.5,
                        num_ants=10, num_iterations=100, alpha=1.0, beta=5.0, rho=0.1, Q=100):
    """
    Multi-objective Ant Colony Optimization for minimizing both travel time and congestion.
    """
    n = time_matrix.shape[0]                              # Number of nodes
    combined_matrix = combine_costs(time_matrix, vehicle_matrix, w_time, w_cong)  # Weighted cost

    with np.errstate(divide='ignore', invalid='ignore'):  # Avoid warnings for division by zero
        visibility = 1.0 / combined_matrix                # Higher visibility = lower cost
        visibility[~np.isfinite(visibility)] = 0.0        # Set invalid entries to zero

    pheromone = initialize_pheromone(n)                   # Start with uniform pheromone
    pareto_front = []                                     # Stores non-dominated solutions

    for _ in range(num_iterations):                       # Repeat for given iterations
        iteration_solutions = []                          # Solutions found in this iteration

        for _ in range(num_ants):                         # Each ant builds a tour
            tour = construct_tour(n, pheromone, visibility, alpha, beta)
            time_cost = tour_length(tour, time_matrix)    # Compute travel time cost
            cong_cost = tour_length(tour, vehicle_matrix) # Compute congestion cost

            if np.isfinite(time_cost) and np.isfinite(cong_cost):   # Only if both valid
                iteration_solutions.append((tour, (time_cost, cong_cost)))
                local_pheromone_update(pheromone, tour, rho)        # Apply local update

        # Update Pareto front with solutions from this iteration
        for new_tour, new_cost in iteration_solutions:
            dominated = False
            pareto_front = [(t, c) for t, c in pareto_front if not dominates(new_cost, c)]  # Remove dominated
            for _, c in pareto_front:
                if dominates(c, new_cost):               # If existing solution dominates new one
                    dominated = True
                    break
            if not dominated:
                pareto_front.append((new_tour, new_cost)) # Add new solution

        # Update pheromone globally using the best solution found so far (lowest sum of objectives)
        if pareto_front:
            best_tour, best_cost = min(pareto_front, key=lambda x: sum(x[1]))
            global_pheromone_update(pheromone, best_tour, sum(best_cost), Q)

    return pareto_front

# --------- CSV PROCESSING ---------

def process_static(timestamp_filter=None):
    """
    Process static traffic metadata and individual traffic data files to build
    time and congestion cost matrices.
    timestamp_filter - optional string to filter by specific time (e.g. '11:30:00')
    """
    metaData = pd.read_csv('trafficMeta.csv')             # Load metadata file
    report_folder = 'traffic_feb_june'                    # Folder with traffic CSV files

    # Create sorted list of all unique nodes
    node_names = sorted(set(metaData['POINT_1_NAME']) | set(metaData['POINT_2_NAME']))
    node_index = {name: idx for idx, name in enumerate(node_names)}  # Map names to indices
    n = len(node_names)

    # Map each node name to its city
    node_to_city = {}
    for _, row in metaData.iterrows():
        node_to_city[row['POINT_1_NAME']] = row['POINT_1_CITY']
        node_to_city[row['POINT_2_NAME']] = row['POINT_2_CITY']

    # Initialize cost matrices with infinity (unreachable)
    time_matrix = np.full((n, n), np.inf)
    vehicle_matrix = np.full((n, n), np.inf)

    # Process each traffic segment
    for _, row in metaData.iterrows():
        report_id = str(row['REPORT_ID'])                  # Unique ID for CSV file
        distance = row['DISTANCE_IN_METERS']               # Distance between nodes in meters
        src, dst = row['POINT_1_NAME'], row['POINT_2_NAME']

        file_path = os.path.join(report_folder, f"trafficData{report_id}.csv")
        if not os.path.exists(file_path):                   # Skip missing files
            continue

        try:
            traffic_df = pd.read_csv(file_path)             # Load traffic data file

            # Ensure required columns exist
            if 'TIMESTAMP' not in traffic_df.columns or 'avgSpeed' not in traffic_df.columns:
                continue

            # Apply timestamp filter if provided
            if timestamp_filter:
                traffic_df = traffic_df[traffic_df['TIMESTAMP'].str.contains(timestamp_filter)]

            if traffic_df.empty:                            # Skip if no data after filtering
                continue

            # Compute mean speed and mean vehicle count
            avg_speed = traffic_df['avgSpeed'].mean()
            vehicle_count = traffic_df['vehicleCount'].mean()

            # Calculate travel time (hours)
            if pd.isna(avg_speed) or avg_speed <= 0:
                travel_time = np.inf
            else:
                distance_km = distance / 1000
                travel_time = distance_km / avg_speed

            # Handle invalid vehicle counts
            if pd.isna(vehicle_count) or vehicle_count < 0:
                vehicle_count = np.inf

            # Update matrices with both directions (undirected)
            s_idx, d_idx = node_index[src], node_index[dst]
            time_matrix[s_idx][d_idx] = travel_time
            time_matrix[d_idx][s_idx] = travel_time
            vehicle_matrix[s_idx][d_idx] = vehicle_count
            vehicle_matrix[d_idx][s_idx] = vehicle_count

        except Exception as e:                               # Handle read/parse errors
            print(f"Error processing file {file_path}: {e}")
            continue

    return time_matrix, vehicle_matrix, node_names, node_to_city

# --------- MAIN EXECUTION ---------

if __name__ == "__main__":
    # Load matrices using traffic data at 11:30 AM
    time_matrix, vehicle_matrix, node_names, node_to_city = process_static(timestamp_filter="11:30:00")

    # Replace infinite values with large constants to make algorithm work
    max_time = np.nanmax(time_matrix[np.isfinite(time_matrix)])
    time_matrix = np.where(np.isinf(time_matrix), max_time * 2, time_matrix)
    max_cong = np.nanmax(vehicle_matrix[np.isfinite(vehicle_matrix)])
    vehicle_matrix = np.where(np.isinf(vehicle_matrix), max_cong * 2, vehicle_matrix)

    # Run multi-objective ACO
    pareto_front = ACO_multi_objective(time_matrix, vehicle_matrix)

    # Display results
    print("\n Pareto Optimal Tours (Multiple Solutions):")
    for i, (tour, (time_cost, cong_cost)) in enumerate(pareto_front, 1):
        print(f"\nSolution {i}:")
        print("  Nodes:  ", [node_names[i] for i in tour])
        print("\nCities: ", [node_to_city.get(node_names[i], 'Unknown') for i in tour])
        print(f"\nTime: {time_cost:.2f} hrs, Congestion: {cong_cost:.2f} units")

    # Extract metrics for possible plotting
    times = [c[0] for _, c in pareto_front]
    congs = [c[1] for _, c in pareto_front]
