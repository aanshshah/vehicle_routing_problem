import random 
from utils import calculate_distance
from strategy import Strategy
import numpy as np
import copy

class Combined_Evo_TwoOpt(Strategy):
	def __init__(self, instance, greedy=False, seed=0, epsilon=1):
		super().__init__(instance)
		random.seed(seed)
		self.greedy = greedy
		self.name = "iterative_sol_combine"
		self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
		self.epsilon = epsilon

	def get_initial_solution(self,seed):
		random.seed(seed)
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

	def approach(self,pop_size=25,top_k=5):
		population = [self.flatten(self.get_initial_solution(i)) for i in range(pop_size)]

		for step in range(1500):
			# print(step)

			population = self.evolve(population,pop_size,top_k,step)
			# if step % 100 == 0:
				# population = [self.iterate_on_2optSwap(p,iterations=10,stop_if_no_progress=True)[0] for p in population]

			if step % 10 == 0: print("Step: {}, {}".format(step,self.calculate_total_distance(population[0])))

		print(population[0])
		return self.unflatten(population[0]), self.calculate_total_distance(population[0]) 

	def evolve(self,population,pop_size,top_k,step,p=None):
		fitness = [(s,self.calculate_total_distance(s)+1000000*(not self.check_within_capacity(s))) for s in population]

		ranked_pop = sorted(fitness,key=lambda x: x[-1])[:top_k]
		if step % 100 == 0:
			ranked_pop = [(self.iterate_on_2optSwap(p,iterations=1,stop_if_no_progress=True)[0],s) for p,s in ranked_pop]



		# print("Best",ranked_pop[0][1])

		new_pop = []
		for x in range(pop_size//(top_k*top_k)):
			for i in range(top_k):
				for j in range(top_k):
					if i != j:
						candidate = self.recombine(ranked_pop[i][0],ranked_pop[j][0])
					else:
						candidate = ranked_pop[i][0]
					# candidate = self.mutate(candidate,p)
					if not self.check_within_capacity(candidate):
						candidate = ranked_pop[i][0]

					new_pop.append(candidate)

		return new_pop

	def mutate(self,solution,p):
		if np.random.rand() < p:
			swaps = self.get_all_neighbors(solution)  #Can't do multiple mutations simultaneosly?
			random.shuffle(swaps)
			choice = swaps[0]
			solution= self.apply_swap(solution,choice)
		return solution


	def recombine(self,solution_1,solution_2,p=0.08): #Should i use numpy arrays??? (This applies to the swap approach too)
		# segment_length = 

	



		# truck_selected = np.random.randint(0,len(solution_1))


		# print(solution_1[truck_selected])


		# if len(solution_1[truck_selected]) <= 2:
		# 	return copy.deepcopy(solution_2)


		# if len(solution_1[truck_selected]) == 3:
		# 	start = 1
		# 	end = 2
		# else:

		start = np.random.randint(1,len(solution_1)-1)
		end = min(np.random.randint(start+1,len(solution_1)),start+solution_1[start:].index(-1))

		# print(start,end)

		segment = solution_1[start:end].copy()

		if np.random.rand() < p:
			segment.reverse()

		# print(segment)


		# print(segment)
		# old_solution = solution_2
		solution_2 = copy.deepcopy(solution_2)

		# for t in range(len(solution_2)):  #Can make this faster
		for c in segment:
			# if c != -1:
			# print(c)
			solution_2.remove(c)



	
		insertion_location = np.random.randint(1,len(solution_2))
		# print(solution_2[truck_added] )
		# print(insertion_location)


		solution_2= solution_2[:insertion_location]+segment + solution_2[insertion_location:]
		# print(solution_2[truck_added])

#total_truck_demand = sum([self.customer_info[c][1] for c in solution[new_truck][1:-1]])
# 			if self.vehicle_capacity >= total_truck_demand:
		# all_ele = []
		# for t in solution_2:
		# 	all_ele+=t

		# print("LEN:",len(set(all_ele)))
		# if len(set(all_ele)) < 101: exit()
		# exit()
		# exit()
		# exit()
		return solution_2


	def flatten(self,all_truck_paths):
		flattened_solution = [-1]
		for truck_locations in all_truck_paths:
			flattened_solution+= truck_locations[1:-1]
			flattened_solution+=[-1]
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

	def iterate_on_2optSwap(self,truck_paths,iterations=1,stop_if_no_progress=True):
		#For every vehicle, for every customer, for every location

		solution = truck_paths
		objective_value = self.calculate_total_distance(solution)
		# print(self.check_within_capacity(solution))
		# exit()
		# print("initial:",objective_value)
		for step in range(1):
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
			# if step % 1 == 0: print("step: {}, cost: {}".format(step,objective_value))

			if stop_if_no_progress:
				if previous_value == objective_value: break;

		return solution,objective_value


	# def iterate_on_2optSwap(self,truck_paths,selector,iterations=1000,stop_if_no_progress=True):
	# 		#For every vehicle, for every customer, for every location

	# 		solution = truck_paths
	# 		objective_value = self.calculate_total_distance(solution)
	# 		print(self.check_within_capacity(solution))
	# 		# exit()
	# 		print("initial:",objective_value)
	# 		for step in range(iterations):
	# 			# print(solution)
	# 			previous_value = objective_value

	# 			for i in range(1,len(solution)-1):
	# 				for j in range(1,len(solution)-1):
	# 					flipped = solution[i:j+1].copy()
	# 					flipped.reverse()
	# 					new_route = solution[0:i] + flipped+solution[j+1:]
	# 					value = self.calculate_total_distance(new_route)
	# 					if self.check_within_capacity(new_route): #self.check_within_capacity(new_route):
	# 						if value < objective_value:
	# 							solution = new_route
	# 							objective_value = value
	# 						# print("good")
	# 					# else:
	# 					# 	print("bad")

	# 			# objective_value = self.calculate_total_distance(solution)
	# 			if step % 1 == 0: print("step: {}, cost: {}".format(step,objective_value))

	# 			if stop_if_no_progress:
	# 				if previous_value == objective_value: break;

	# 		return solution,objective_value







