import random 
from utils import calculate_distance
from strategy import Strategy
import numpy as np
import copy

class TwoOpt_Solution(Strategy):
	def __init__(self, instance, greedy=False, seed=0, epsilon=1):
		super().__init__(instance)
		random.seed(seed)
		self.greedy = greedy
		self.name = "iterative_sol_2opt"
		self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
		self.epsilon = epsilon

	def get_initial_solution(self):
		# random.seed(seed)
		def random_search():
			if self.greedy:
				self.customer_info.sort(key=lambda x: x[1])
			else:
				random.shuffle(self.customer_info)
			visited = set()
			total_distance_traveled = 0
			all_truck_paths = []
			for i in range(self.num_vehicles): 
				capacity = self.vehicle_capacity 
				truck_path = [0]
				for customer_idx, customer_demand, _, _ in self.customer_info:
					if customer_idx == 0: continue
					if customer_idx in visited:
						continue		
					if capacity - customer_demand < 0:
						continue
					visited.add(customer_idx)
					capacity -= customer_demand
					truck_path.append(customer_idx)
					if capacity == 0: break 
				truck_path.append(0) #goes back to depot
				all_truck_paths.append(truck_path)
			return all_truck_paths

		while True:
			self.attempts += 1
			visited = set()
			all_truck_paths = random_search()
			for path in all_truck_paths:
				for customer_idx in path:
					visited.add(customer_idx)
			if len(visited) == self.num_customers:
				return all_truck_paths



	def get_initial_solution(self):
		def random_search():
			ordering = self.customer_info.copy()
			if self.greedy:
				self.customer_info.sort(key=lambda x: x[1])
			else:
				random.shuffle(ordering)
			visited = set()
			total_distance_traveled = 0
			all_truck_paths = []
			for i in range(self.num_vehicles): 
				capacity = self.vehicle_capacity 
				truck_path = [0]
				for customer_idx, customer_demand, _, _ in ordering:
					if customer_idx == 0: continue

					if customer_idx in visited:
						continue		
					if capacity - customer_demand < 0:
						continue
					visited.add(customer_idx)
					capacity -= customer_demand
					truck_path.append(customer_idx)
					if capacity == 0: break 
				truck_path.append(0) #goes back to depot
				all_truck_paths.append(truck_path)
			return all_truck_paths

		while True:
			self.attempts += 1
			visited = set()
			all_truck_paths = random_search()
			for path in all_truck_paths:
				for customer_idx in path:
					visited.add(customer_idx)
			if len(visited) == self.num_customers:
				return all_truck_paths

	def approach(self):
		all_truck_paths = self.get_initial_solution()
		flattened_solution = self.flatten(all_truck_paths)
		
		
		all_truck_paths, total_distance_traveled =  self.iterate_on_2optSwap(flattened_solution,iterations=1000,selector=greedy)

		return self.unflatten(all_truck_paths), total_distance_traveled 

	def flatten(self,all_truck_paths):
		flattened_solution = []
		for truck_locations in all_truck_paths:
			flattened_solution+=[-1]
			flattened_solution+= truck_locations[1:-1]
			
		return flattened_solution

	def unflatten(self,flattened):
		all_truck_paths = []

		for l in flattened:
			if l == -1:
				if len(all_truck_paths) != 0:
					all_truck_paths[-1].append(0)
				all_truck_paths.append([0])
			else:
				all_truck_paths[-1].append(l)

		all_truck_paths[-1].append(0)

		return all_truck_paths







	def apply_swap(self,solution,swap):
		starting_index = swap[0]
		ending_index = swap[1]

		ele_to_move = solution[starting_index]
		del solution[starting_index]
		solution.insert(ending_index,ele_to_move)

		return solution

	def undo_swap(self,solution,swap):
		starting_index = swap[0]
		ending_index = swap[1]
		ele_to_move = solution[ending_index]
		del solution[ending_index]
		solution.insert(starting_index,ele_to_move)		
		return solution

	def get_all_neighbors(self,truck_paths_flat):
		swaps = []

		for i in truck_paths_flat[1:]:
			for j in truck_paths_flat[1:]:
				swaps.append((i,j))

		return swaps

	def evaluate_neighbors(self,truck_paths,sample_neighbors=False,sample_size=-1):
		solution = truck_paths
		swaps = self.get_all_neighbors(truck_paths)

		if sample_neighbors:
			random.shuffle(swaps)
			swaps = swaps[:sample_size]

		evaluated_swaps = []
		for s in swaps:
			solution = self.apply_swap(solution,s)
			# total_truck_demand = sum([self.customer_info[c][1] for c in solution[new_truck][1:-1]])
			# if self.vehicle_capacity >= total_truck_demand:
			if self.check_within_capacity(solution):
				score = self.calculate_total_distance(solution)
			else: score = np.Inf
			evaluated_swaps.append((s,score))
			self.undo_swap(solution,s)
		assert truck_paths == solution
		return evaluated_swaps



	# def iterate_on_solution(self,truck_paths,selector,iterations=1000,stop_if_no_progress=False):
	# 		#For every vehicle, for every customer, for every location

	# 		solution = truck_paths
	# 		objective_value = self.calculate_total_distance(solution)
	# 		print("initial:",objective_value)
	# 		for step in range(iterations):
	# 			# print(solution)
	# 			previous_value = objective_value
	# 			swaps = self.evaluate_neighbors(solution,sample_neighbors=True,sample_size=100)
	# 			chosen = selector(swaps,previous_value,step,iterations)
	# 			if chosen is not None:
	# 				solution = self.apply_swap(solution,chosen)
	# 			objective_value = self.calculate_total_distance(solution)
	# 			if step % 100 == 0: print("step: {}, cost: {}".format(step,objective_value))

	# 			if stop_if_no_progress:
	# 				if previous_value == objective_value: break;

	# 		return solution,objective_value

	def check_within_capacity(self,solution_flat):
		total_d = 0
		for l in solution_flat:
			if l == -1: total_d = 0
			else:
				total_d+=self.customer_info[l][1]

			# print(total_d)

			if total_d > self.vehicle_capacity:
				return False

		return True

	def calculate_total_distance(self,solution_flat):
		total_distance_traveled = 0

		_,_,start_x,start_y = self.customer_info[0]

		for l in solution_flat:
			if l == -1:
				_,_,end_x,end_y = self.customer_info[0]
			else:
				_,_,end_x,end_y = self.customer_info[l]

			total_distance_traveled += calculate_distance(start_x, start_y, end_x, end_y)

			start_x = end_x
			start_y = end_y

		return total_distance_traveled


	def iterate_on_2optSwap(self,truck_paths,selector,iterations=1000,stop_if_no_progress=True):
			#For every vehicle, for every customer, for every location

			solution = truck_paths
			objective_value = self.calculate_total_distance(solution)
			print(self.check_within_capacity(solution))
			# exit()
			print("initial:",objective_value)
			for step in range(iterations):
				# print(solution)
				previous_value = objective_value

				for i in range(1,len(solution)-1):
					for j in range(1,len(solution)-1):
						flipped = solution[i:j+1].copy()
						flipped.reverse()
						new_route = solution[0:i] + flipped+solution[j+1:]
						value = self.calculate_total_distance(new_route)
						if self.check_within_capacity(new_route): #self.check_within_capacity(new_route):
							if value < objective_value:
								solution = new_route
								objective_value = value
							# print("good")
						# else:
						# 	print("bad")

				# objective_value = self.calculate_total_distance(solution)
				if step % 1 == 0: print("step: {}, cost: {}".format(step,objective_value))

				if stop_if_no_progress:
					if previous_value == objective_value: break;

			return solution,objective_value

	# def check_within_capacity(self,solution):
	# 	for truck in solution:
	# 		total_truck_demand = sum([self.customer_info[c][1] for c in truck[1:-1]])
	# 		if total_truck_demand > self.vehicle_capacity:
	# 			return False
		
	# 	return True

def greedy(swaps,previous_value=None,step=None,max_step=None):
		greedy_best = min(swaps,key=lambda x:x[-1])
		chosen,value = greedy_best
		return chosen

def simmulated_annealing(swaps,previous_value,step,max_step):
	sample,value = swaps[0]

	if value < previous_value:
		return sample
	else:
		T = 100*np.exp(-0.001*(step+1/max_step)) #Temp function. Neeed to tune
		if step % 1000 == 0: print(T)
		accept_P = np.exp(-(value-previous_value)/T) #(Kirkpatrick et al.,)
		# print(accept_P)
		if accept_P >= np.random.rand():
			return sample
		else: return None





# procedure 2optSwap(route, i, k) {
#     1. take route[0] to route[i-1] and add them in order to new_route
#     2. take route[i] to route[k] and add them in reverse order to new_route
#     3. take route[k+1] to end and add them in order to new_route
#     return new_route;
# }
# Here is an example of the above with arbitrary input:

# Example route: A → B → C → D → E → F → G → H → A
# Example parameters: i = 4, k = 7 (starting index 1)
# Contents of new_route by step:
# (A → B → C)
# A → B → C → (G → F → E → D)
# A → B → C → G → F → E → D → (H → A)
# This is the complete 2-opt swap making use of the above mechanism:

# repeat until no improvement is made {
#     start_again:
#     best_distance = calculateTotalDistance(existing_route)
#     for (i = 1; i <= number of nodes eligible to be swapped - 1; i++) {
#         for (k = i + 1; k <= number of nodes eligible to be swapped; k++) {
#             new_route = 2optSwap(existing_route, i, k)
#             new_distance = calculateTotalDistance(new_route)
#             if (new_distance < best_distance) {
#                 existing_route = new_route
#                 best_distance = new_distance
#                 goto start_again
#             }
#         }
#     }
# }