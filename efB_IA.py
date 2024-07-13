import numpy as np
import heapq
import time
from instances import instances

class StationOptimization:
    def __init__(self, family_distribution):
        self.family_distribution = np.array(family_distribution)
        self.N, self.M = self.family_distribution.shape
        self.generations = 0
        self.evaluations = 0
        self.max_stations = 5
        self.memo = {}

    def get_cost(self, distance):
        return {
            0: 0,
            1: 0,
            2: 1,
            3: 2,
            4: 4,
            5: 8
        }.get(distance, 10)

    def heuristic(self, state):
        if not state:
            return float('inf')

        total_travel_cost = 0
        total_families = 0

        for x in range(self.N):
            for y in range(self.M):
                if self.family_distribution[x, y] > 0:
                    nearest_station_dist = min((max(abs(x - sx), abs(y - sy)) for sx, sy in state), default=10)
                    total_families += self.family_distribution[x, y]
                    cost = self.get_cost(nearest_station_dist)
                    total_travel_cost += cost * self.family_distribution[x, y]

        A = len(state)
        B = total_travel_cost / total_families if total_families != 0 else float('inf')

        if B >= 3.2 and len(state) > 2:
            return float('inf')

    
        if B >= 5 and len(state) <= 2:
            return float('inf')

        return 1000 * A + 100 * B 

    def cost(self, state):
        state_tuple = tuple(sorted(state))
        if state_tuple in self.memo:
            return self.memo[state_tuple]

        total_travel_cost = 0
        total_families = 0

        for x in range(self.N):
            for y in range(self.M):
                if self.family_distribution[x, y] > 0:
                    nearest_station_dist = min((max(abs(x - sx), abs(y - sy)) for sx, sy in state), default=20)
                    total_families += self.family_distribution[x, y]
                    total_travel_cost += self.get_cost(nearest_station_dist) * self.family_distribution[x, y]

        A = len(state)
        B = total_travel_cost / total_families if total_families != 0 else float('inf')

        cost_value = 1000 * A + 100 * B
        self.memo[state_tuple] = cost_value
        
        self.evaluations += 1
        print(f"Avaliações: {self.evaluations}, Gerações: {self.generations}")

        return cost_value

    def get_neighbors(self, state):
        neighbors = []
        sorted_families = sorted(
            ((x, y) for x in range(self.N) for y in range(self.M) if 0 < x < self.N-1 and 0 < y < self.M-1), 
            key=lambda pos: -self.family_distribution[pos[0], pos[1]]
        )

        limit = 190
        for i, (x, y) in enumerate(sorted_families):
            if i >= limit:
                break
            if (x, y) not in state:
                new_state = set(state)
                new_state.add((x, y))
                if len(new_state) <= self.max_stations:
                    neighbors.append(new_state)
            else:
                new_state = set(state)
                new_state.remove((x, y))
                if len(new_state) > 0:
                    neighbors.append(new_state)
        return neighbors

    def a_star(self):
        max_generations = 1000000
        max_time = 60

        start_time = time.time()
        best_solutions = {}

        frontier = []
        initial_state = set()
        heapq.heappush(frontier, (self.heuristic(initial_state), initial_state))
        cost_so_far = {tuple(initial_state): 0}
        explored = set()

        while frontier:
            current_time = time.time()
            self.generations += 1
            if self.generations >= max_generations or current_time - start_time >= max_time:
                print("Time limit or number of generations reached.")

                break

            _, current = heapq.heappop(frontier)
            current_tuple = tuple(sorted(current))

            if current_tuple in explored:
                continue
            explored.add(current_tuple)

            

            current_cost = self.cost(current)
            total_travel_cost = sum(
                self.get_cost(min((max(abs(x - sx), abs(y - sy)) for sx, sy in current), default=20)) * self.family_distribution[x, y]
                for x in range(self.N) for y in range(self.M) if self.family_distribution[x, y] > 0
            )
            total_families = sum(self.family_distribution[x, y] for x in range(self.N) for y in range(self.M))
            B = total_travel_cost / total_families if total_families != 0 else float('inf')

            num_stations = len(current)
            if B < 3 and (num_stations not in best_solutions or B < best_solutions[num_stations][2]):
                best_solutions[num_stations] = (current, current_cost, B, self.generations, current_time - start_time, self.evaluations)

            neighbors = self.get_neighbors(current)
            for next_state in neighbors:
                next_tuple = tuple(sorted(next_state))
                new_cost = self.cost(next_state)
                priority = new_cost + self.heuristic(next_state)
                if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                    cost_so_far[next_tuple] = new_cost
                    heapq.heappush(frontier, (priority, next_state))

        return best_solutions

    def visualize_map(self, stations):
        color_map = {
            0: "\033[34m",
            1: "\033[32m",
            2: "\033[33m",
            3: "\033[35m",
            4: "\033[31m",
            5: "\033[36m",
            6: "\033[37m",
            7: "\033[30m",
        }

        for i in range(self.N):
            for j in range(self.M):
                if (i, j) in stations:
                    print(f"\033[34m{self.family_distribution[i][j]:2}\033[0m", end=" ")
                else:
                    nearest_station_dist = min((max(abs(i - sx), abs(j - sy)) for sx, sy in stations), default=20)
                    dist = min(7, int(nearest_station_dist))
                    print(f"{color_map[dist]}{self.family_distribution[i][j]:2}\033[0m", end=" ")
            print()

def menu():
    while True:
        print("Choose an option:")
        print("1. Calculate solution for a specific instance")
        print("2. Calculate solutions for all instances")
        print("3. Calculate solutions for a range of instances")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")


        if choice == '1':
            print(f"There's {len(instances)} instances available.")
            index = int(input(f"Choose one instance (1-{len(instances)}): ")) - 1
            if 0 <= index < len(instances):
                family_distribution = instances[index]
                optimizer = StationOptimization(family_distribution)
                best_solutions = optimizer.a_star()
                print(f"Generations: {optimizer.generations}")
                for num_stations in sorted(best_solutions.keys()):
                    stations, total_cost, B, generations, time_spent, evaluations = best_solutions[num_stations]
                    print(f"Total stations: {num_stations}, Total Cost: {total_cost}, Average Travel Cost: {B}, Stations: {stations}, Time: {time_spent:.2f}s, Generations: {generations}, Evaluations: {evaluations}")
                    optimizer.visualize_map(stations)
                    print()
            else:
                print("Invalid index.")

        elif choice == '2':
            for i, family_distribution in enumerate(instances):
                print(f"Calculating solution for instance {i + 1}")
                optimizer = StationOptimization(family_distribution)
                best_solutions = optimizer.a_star()
                print(f"Generations: {optimizer.generations}")
                for num_stations in sorted(best_solutions.keys()):
                    stations, total_cost, B, generations, time_spent, evaluations = best_solutions[num_stations]
                    print(f"Total stations: {num_stations}, Total Cost: {total_cost}, Average Travel Cost: {B}, Stations: {stations}, Time: {time_spent:.2f}s, Generations: {generations}, Evaluations: {evaluations}")
                    optimizer.visualize_map(stations)
                    print()
        elif choice == '3':
            start_index = int(input(f"Choose the start index of the instance (1-{len(instances)}): ")) - 1
            end_index = int(input(f"Choose the end index of the instance (1-{len(instances)}): ")) - 1

            if 0 <= start_index <= end_index < len(instances):
                for i in range(start_index, end_index + 1):
                    print(f"Calculating solution for instance {i + 1}")
                    family_distribution = instances[i]
                    optimizer = StationOptimization(family_distribution)
                    best_solutions = optimizer.a_star()
                    print(f"Generations: {optimizer.generations}")
                    for num_stations in sorted(best_solutions.keys()):
                        stations, total_cost, B, generations, time_spent, evaluations = best_solutions[num_stations]
                        print(f"Total stations: {num_stations}, Total Cost: {total_cost}, Average Travel Cost: {B}, Stations: {stations}, Time: {time_spent:.2f}s, Generations: {generations}, Evaluations: {evaluations}")
                        optimizer.visualize_map(stations)
                        print()
            else:
                print("Invalid Index")
        elif choice == '4':
            break
        else:
            print("Invalid choice. Try again (1-4)")

if __name__ == "__main__":
    menu()
