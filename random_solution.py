import random 
from utils import calculate_distance
from strategy import Strategy

class RandomStrategy(Strategy):
	def __init__(self, instance):
		super().__init__(instance)

	def approach(self):
		def random_search():
			truck_x, truck_y = self.depo_coords
			random.shuffle(self.customer_info)
			visited = set()
			total_distance_traveled = 0
			all_truck_paths = []
			for i in range(self.num_vehicles): 
				capacity = self.vehicle_capacity 
				truck_path = [0]
				for customer_idx, customer_demand, customer_x, customer_y in self.customer_info:
					if customer_idx in visited:
						continue		
					if capacity - customer_demand < 0:
						continue
					visited.add(customer_idx)
					capacity -= customer_demand
					truck_path.append(customer_idx)
					total_distance_traveled += calculate_distance(truck_x, truck_y, customer_x, customer_y)
					truck_x = customer_x
					truck_y = customer_y
					if capacity == 0: break 
				truck_path.append(0) #goes back to depot
				all_truck_paths.append(truck_path)
			return all_truck_paths, total_distance_traveled

		while True:
			self.attempts += 1
			visited = set()
			all_truck_paths, total_distance_traveled = random_search()
			for path in all_truck_paths:
				for customer_idx in path:
					visited.add(customer_idx)
			if len(visited) == self.num_customers:
				return all_truck_paths, total_distance_traveled

	def run(self):
		self.paths, self.distance = self.approach()
		return self.paths, self.distance