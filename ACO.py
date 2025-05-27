# Ant Colony Optimization (ACO) for Travelling Salesman Problem (TSP)
import numpy as np
import random

def distance_matrix(cities):
    n = len(cities)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
            else:
                matrix[i][j] = np.inf
    return matrix

def initialize_pheromone(n):
    return np.ones((n, n))

def select_next_city(current_city, unvisited, pheromone, visibility, alpha, beta):
    pheromone_power = np.power(pheromone[current_city][unvisited], alpha)
    visibility_power = np.power(visibility[current_city][unvisited], beta)
    probabilities = pheromone_power * visibility_power
    probabilities /= np.sum(probabilities)
    return np.random.choice(unvisited, p=probabilities)

def construct_tour(n, pheromone, visibility, alpha, beta):
    tour = []
    unvisited = list(range(n))
    current_city = random.choice(unvisited)
    tour.append(current_city)
    unvisited.remove(current_city)
    while unvisited:
        next_city = select_next_city(current_city, unvisited, pheromone, visibility, alpha, beta)
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    return tour

def local_pheromone_update(pheromone, tour, rho):
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        pheromone[a][b] *= (1 - rho)
        pheromone[b][a] = pheromone[a][b]

def global_pheromone_update(pheromone, best_tour, best_length, Q):
    for i in range(len(best_tour)):
        a, b = best_tour[i], best_tour[(i + 1) % len(best_tour)]
        pheromone[a][b] += Q / best_length
        pheromone[b][a] = pheromone[a][b]

def tour_length(tour, dist_matrix):
    return sum(dist_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def ACO(cities, num_ants=10, num_iterations=100, alpha=1.0, beta=5.0, rho=0.1, Q=100):
    n = len(cities)
    dist_matrix = distance_matrix(cities)
    visibility = 1 / dist_matrix
    pheromone = initialize_pheromone(n)
    best_tour = None
    best_length = float('inf')

    for _ in range(num_iterations):
        all_tours = []
        for _ in range(num_ants):
            tour = construct_tour(n, pheromone, visibility, alpha, beta)
            all_tours.append(tour)
            local_pheromone_update(pheromone, tour, rho)

        iteration_best = min(all_tours, key=lambda t: tour_length(t, dist_matrix))
        iteration_length = tour_length(iteration_best, dist_matrix)

        if iteration_length < best_length:
            best_tour = iteration_best
            best_length = iteration_length

        global_pheromone_update(pheromone, best_tour, best_length, Q)

    return best_tour, best_length

# Example usage:
cities = [(9,0), (10,11), (90,100), (900,40), (78,67), (88,69), (0,10), (66,55), (91,23), (78,32)]
tour, cost = ACO(cities)
print("Best Tour:", [int(city) for city in tour])
print("Tour Cost:", cost)
