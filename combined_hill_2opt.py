import random 
from utils import calculate_distance
from strategy import Strategy
import numpy as np
import copy

class Combined_Hill2OPT_TwoOpt(Strategy):
	def __init__(self, instance, greedy=False, seed=0, epsilon=1):
		super().__init__(instance)
		random.seed(seed)
		# self.greedy = greedy
		self.name = "iterative_sol_combine_hill_2opt"
		self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
		self.epsilon = epsilon

	def get_initial_solution(self):
		# random.seed(seed)
		def random_search():

			ordering = self.customer_info.copy()
			# if self.greedy:
			# 	self.customer_info.sort(key=lambda x: x[1])
			# else:
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

	

	def iterate_on_2optSwap(self,truck_paths,iterations=100,stop_if_no_progress=True):
		#For every vehicle, for every customer, for every location

		# print(truck_paths.count(-1))

		def check_within_capacity(solution_flat):
			total_d = 0
			for l in solution_flat:
				if l == -1: total_d = 0
				else:
					total_d+=self.customer_info[l][1]

				# print(total_d)

				if total_d > self.vehicle_capacity:
					return False

			return True

		def calculate_total_distance(solution_flat):
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

		solution = truck_paths
		objective_value = calculate_total_distance(solution)
		# print(self.check_within_capacity(solution))
		# exit()
		# print("initial:",objective_value)
		for step in range(iterations):
			# print(solution)
			previous_value = objective_value

			for i in range(1,len(solution)-1):
				for j in range(i+1,len(solution)-1):
					flipped = solution[i:j+1].copy()
					flipped.reverse()
					new_route = solution[0:i] + flipped+solution[j+1:]
					value = calculate_total_distance(new_route)
					if check_within_capacity(new_route): #self.check_within_capacity(new_route):
						if value < objective_value:
							solution = new_route
							objective_value = value
						# print("good")
					# else:
					# 	print("bad")

			# objective_value = self.calculate_total_distance(solution)
			# if step % 1 == 0: print("step: {}, cost: {}".format(step,objective_value))

			if stop_if_no_progress:
				if previous_value == objective_value: break;

		return solution,objective_value\

	def approach(self):
		all_truck_paths = self.get_initial_solution()
		all_truck_paths, total_distance_traveled =  self.iterate_on_solution(all_truck_paths,iterations=1000,selector=greedy)
		# all_truck_paths, total_distance_traveled = self.iterate_on_solution(all_truck_paths,iterations=10000,selector=simmulated_annealing)

		return all_truck_paths, total_distance_traveled 

	def calculate_total_distance(self,solution):
		total_distance_traveled = 0
		for truck_locations in solution:
			for start,end in zip(truck_locations[:-1],truck_locations[1:]):
				_,_,start_x,start_y = self.customer_info[start]
				_,_,end_x,end_y = self.customer_info[end]

				total_distance_traveled += calculate_distance(start_x, start_y, end_x, end_y)
		return total_distance_traveled
					

	def apply_swap(self,solution,swap):
		starting_index = swap[0]
		ending_index = swap[1]

		ele_to_move = solution[starting_index[0]][starting_index[1]]
		del solution[starting_index[0]][starting_index[1]]
		solution[ending_index[0]].insert(ending_index[1],ele_to_move)

		return solution,ending_index[0]

	def undo_swap(self,solution,swap):
		starting_index = swap[0]
		ending_index = swap[1]
		ele_to_move = solution[ending_index[0]][ending_index[1]]
		del solution[ending_index[0]][ending_index[1]]
		solution[starting_index[0]].insert(starting_index[1],ele_to_move)		
		return solution

	def get_all_neighbors(self,truck_paths):
		swaps = []

		for truck_number,locations in enumerate(truck_paths):
			# print(truck_number)
			if len(locations) < 2: continue
			else:
				for i,l in enumerate(locations[1:-1]):
					index = i + 1
					new_solution = truck_paths #.copy()
					for new_truck_number,truck_locations in enumerate(new_solution):
						for j in range(len(truck_locations)-2):
							swaps.append(((truck_number,index),(new_truck_number,j+1)))
		return swaps

	def evaluate_neighbors(self,truck_paths,sample_neighbors=False,sample_size=-1):
		solution = truck_paths
		swaps = self.get_all_neighbors(truck_paths)

		if sample_neighbors:
			random.shuffle(swaps)
			swaps = swaps[:sample_size]

		evaluated_swaps = []
		for s in swaps:
			solution, new_truck = self.apply_swap(solution,s)
			total_truck_demand = sum([self.customer_info[c][1] for c in solution[new_truck][1:-1]])
			if self.vehicle_capacity >= total_truck_demand:
				score = self.calculate_total_distance(solution)
			else: score = np.Inf
			evaluated_swaps.append((s,score))
			self.undo_swap(solution,s)
		assert truck_paths == solution
		return evaluated_swaps

	def check_within_capacity(self,solution):
		# print(self.customer_info)

		for truck in solution:
			total_truck_demand = sum([self.customer_info[c][1] for c in truck[1:-1]])
			if total_truck_demand > self.vehicle_capacity:
				return False
		
		return True

	def iterate_on_solution(self,truck_paths,selector,iterations=1000,stop_if_no_progress=False):
			#For every vehicle, for every customer, for every location

			solution = truck_paths
			objective_value = self.calculate_total_distance(solution)
			# print(self.check_within_capacity(solution))
			# exit()
			print("initial:",objective_value)
			for step in range(iterations):
				previous_value = objective_value
				swaps = self.evaluate_neighbors(solution,sample_neighbors=True,sample_size=100)
				chosen = selector(swaps,previous_value,step,iterations)
				if chosen is not None:
					solution,_ = self.apply_swap(solution,chosen)
				objective_value = self.calculate_total_distance(solution)

				if step % 100 == 0:
					solution,objective_value = self.iterate_on_2optSwap(self.flatten(solution),iterations=5,stop_if_no_progress=True)
					solution = self.unflatten(solution)


				if step % 100 == 0: print("step: {}, cost: {}".format(step,objective_value))

				if stop_if_no_progress:
					if previous_value == objective_value: break;

			return solution,objective_value


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


