import random 
from utils import calculate_distance
from strategy import Strategy
import copy

class Evolution_Solution(Strategy):
	def __init__(self, instance, greedy=False, seed=0, epsilon=1):
		super().__init__(instance)
		random.seed(seed)
		self.greedy = greedy
		self.name = "iterative_sol_evo"
		self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
		self.epsilon = epsilon

	def get_initial_solution(self,seed):
		random.seed(seed)

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

	def approach(self,iterations=1):
		best = 1e26
		assignment = None
		for i in range(iterations):
			cand_assignment, cand_value = self.approach_run(restart=i)
			if cand_value < best:
				best = cand_value
				assignment = cand_assignment
		return assignment,best


	def approach_run(self,pop_size=25,top_k=5,restart=0):
		population = [self.flatten(self.get_initial_solution(i+restart*pop_size)) for i in range(pop_size)]

		for step in range(10000):
			# print(step)
			population = self.evolve(population,pop_size,top_k)

			if step % 10:
				self.paths = self.unflatten(population[0])
				self.distance = self.calculate_total_distance(population[0]) 
			# if step % 10 == 0: print("Step: {}, {}".format(step,self.calculate_total_distance(population[0])))

		# print(population[0])
		self.paths = self.unflatten(population[0])
		self.distance = self.calculate_total_distance(population[0]) 
		return self.paths, self.distance

	def evolve(self,population,pop_size,top_k,p=None):
		fitness = [(s,self.calculate_total_distance(s)+1e26*(not self.check_within_capacity(s))) for s in population]

		ranked_pop = sorted(fitness,key=lambda x: x[-1])[:top_k]

		# print(ranked_pop)

		new_pop = []
		for x in range(pop_size//(top_k*top_k)):
			for i in range(top_k):
				for j in range(top_k):
					if i != j:
						candidate = self.recombine(ranked_pop[i][0],ranked_pop[j][0])
					else:
						candidate = ranked_pop[i][0]  #Elistism!
					if not self.check_within_capacity(candidate):
						candidate = ranked_pop[i][0]

					new_pop.append(candidate)

		return new_pop

	# def mutate(self,solution,p):
	# 	if np.random.rand() < p:
	# 		swaps = self.get_all_neighbors(solution)  #Can't do multiple mutations simultaneosly?
	# 		random.shuffle(swaps)
	# 		choice = swaps[0]
	# 		solution= self.apply_swap(solution,choice)
	# 	return solution


	def recombine(self,solution_1,solution_2,reverse_p=0.1): #Should i use numpy arrays??? (This applies to the swap approach too)
			
			start = random.randint(1,len(solution_1)-1)
			if -1 in solution_1[start:]:
				max_end = start + solution_1[start:].index(-1)
			else:
				max_end = start + len(solution_1[start:])

			end = random.randint(start+1,max_end) if start+1 < max_end else max_end

		
			segment = solution_1[start:end].copy()

			if random.random() < reverse_p:
				segment.reverse()

			solution_2 = copy.deepcopy(solution_2)

			for c in segment:
				# if c != -1:
				# print(c)
				solution_2.remove(c)

			insertion_location = random.randint(1,len(solution_2))
			solution_2= solution_2[:insertion_location]+segment + solution_2[insertion_location:]
			# print(solution_2)
			return solution_2



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

	# def get_all_neighbors(self,truck_paths_flat):
	# 	swaps = []

	# 	for i in truck_paths_flat[1:]:
	# 		for j in truck_paths_flat[1:]:
	# 			swaps.append((i,j))

	# 	return swaps

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








