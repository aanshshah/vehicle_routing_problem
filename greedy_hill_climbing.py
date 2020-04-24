import random 
from utils import calculate_distance
from strategy import Strategy

class GreedyHillClimbing(Strategy):
	def __init__(self, instance, greedy=False, seed=0, epsilon=1):
		super().__init__(instance)
		random.seed(seed)
		self.greedy = greedy
		self.name = "GreedyHillClimbing"
		self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
		self.epsilon = epsilon
	
	def approach(self):
		def bfs_search():
			truck_x, truck_y = self.depo_coords
			visited = set()
			for i in range(self.num_vehicles):
				truck_visit = set()
				capacity, distance, truck_path = self.vehicle_map[i]
				while len(truck_visit) < self.num_customers:
					next_customer_distance = float("inf")
					next_customer_index = None

					for i in range(self.num_customers):
						customer_idx, customer_demand, customer_x, customer_y = self.customer_info[i]
						if i in truck_visit or i in visited: continue
						if capacity - customer_demand < 0: 
							truck_visit.add(customer_idx)
							continue
						customer_distance = calculate_distance(truck_x, truck_y, customer_x, customer_y)
						if customer_distance < next_customer_distance: #try reversing the order as well 
							next_customer_index = customer_idx
							next_customer_distance = distance

					if not next_customer_index or capacity == 0:
						self.vehicle_map[i] = capacity, distance, truck_path
						break

					capacity -= customer_demand
					distance += customer_distance
					truck_path.add(next_customer_index)
					visited.add(next_customer_index)
					truck_visit.add(next_customer_index)
					truck_x = customer_x
					truck_y = customer_y 
			return visited

			visited = bfs_search()
			if len(visited) < self.num_customers:
				return False
			all_truck_paths = [[] for x in self.num_vehicles]
			total_distance_traveled = 0
			for vehicle_idx, _, distance, truck_path in self.vehicle_map.items():
				all_truck_paths[vehicle_idx] = truck_path
				total_distance_traveled += distance
			return all_truck_paths, total_distance_traveled

import random 
from utils import calculate_distance
from strategy import Strategy

class GreedyHillClimbingEpsilon(Strategy):
	def __init__(self, instance, greedy=False, seed=0, epsilon=1):
		super().__init__(instance)
		random.seed(seed)
		self.greedy = greedy
		self.name = "GreedyHillClimbing"
		self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
		self.epsilon = epsilon
	
	def approach(self):
		def bfs_search():
			truck_x, truck_y = self.depo_coords
			visited = set()
			for i in range(self.num_vehicles):
				truck_visit = set()
				capacity, distance, truck_path = self.vehicle_map[i]
				while len(truck_visit) < self.num_customers:
					next_customer_distance = float("inf")
					next_customer_index = None

					for i in range(self.num_customers):
						customer_idx, customer_demand, customer_x, customer_y = self.customer_info[i]
						if i in truck_visit or i in visited: continue
						if capacity - customer_demand < 0: 
							truck_visit.add(customer_idx)
							continue
						
						customer_distance = calculate_distance(truck_x, truck_y, customer_x, customer_y)
						if customer_distance < next_customer_distance: #try reversing the order as well 
							next_customer_index = customer_idx
							next_customer_distance = distance

					if not next_customer_index or capacity == 0:
						self.vehicle_map[i] = capacity, distance, truck_path
						break

					capacity -= customer_demand
					distance += customer_distance
					truck_path.add(next_customer_index)
					visited.add(next_customer_index)
					truck_visit.add(next_customer_index)
					truck_x = customer_x
					truck_y = customer_y 
			return visited

			visited = bfs_search()
			if len(visited) < self.num_customers:
				return False
			all_truck_paths = [[] for x in self.num_vehicles]
			total_distance_traveled = 0
			for vehicle_idx, _, distance, truck_path in self.vehicle_map.items():
				all_truck_paths[vehicle_idx] = truck_path
				total_distance_traveled += distance
			return all_truck_paths, total_distance_traveled