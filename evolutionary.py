import random 
from utils import calculate_distance
from strategy import Strategy
import numpy as np
import copy

class Evolution_Solution(Strategy):
	def __init__(self, instance, greedy=False, seed=0, epsilon=1):
		super().__init__(instance)
		random.seed(seed)
		self.greedy = greedy
		self.name = "iterative_sol_evo"
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

	def approach(self,pop_size=100,top_k=10):
		population = [self.get_initial_solution() for i in range(pop_size)]

		for step in range(1000):
			print(step)
			population = self.evolve(population,top_k)




		# all_truck_paths, total_distance_traveled =  self.iterate_on_solution(all_truck_paths,iterations=1000,selector=greedy) # Could try doing this periodically
		# all_truck_paths, total_distance_traveled = self.iterate_on_solution(all_truck_paths,iterations=10000,selector=simmulated_annealing)

		return all_truck_paths, total_distance_traveled 

	def evolve(self,population,top_k=10,p=0.30):
		fitness = [(s,self.calculate_total_distance(s)) for s in population]
		ranked_pop = sorted(fitness,key=lambda x: x[-1])[:top_k]
		print("Best",ranked_pop[0][1])

		new_pop = []
		for i in range(top_k):
			for j in range(top_k):
				if i != j:
					candidate = self.recombine(ranked_pop[i][0],ranked_pop[j][0])
				else:
					candidate = ranked_pop[i][0]
				candidate = self.mutate(candidate,p)
				new_pop.append(candidate)

		return new_pop

	def mutate(self,solution,p):
		if np.random.rand() < p:
			swaps = self.get_all_neighbors(solution)  #Can't do multiple mutations simultaneosly?
			random.shuffle(swaps)
			choice = swaps[0]
			solution, _ = self.apply_swap(solution,choice)
		return solution


	def recombine(self,solution_1,solution_2): #Should i use numpy arrays??? (This applies to the swap approach too)
		# segment_length = 

	



		truck_selected = np.random.randint(0,len(solution_1))


		# print(solution_1[truck_selected])


		if len(solution_1[truck_selected]) <= 2:
			return copy.deepcopy(solution_2)


		if len(solution_1[truck_selected]) == 3:
			start = 1
			end = 2
		else:
			start = np.random.randint(1,len(solution_1[truck_selected])-1)
			end = np.random.randint(start+1,len(solution_1[truck_selected]))

		# print(start,end)


		segment = solution_1[truck_selected][start:end]

		# print(segment)


		# print(segment)
		solution_2 = copy.deepcopy(solution_2)

		for t in range(len(solution_2)):  #Can make this faster
			for c in segment:
				if c in solution_2[t]: 
					solution_2[t].remove(c)



		truck_added = np.random.randint(0,len(solution_2))

		# print(len(solution_2[truck_added]))
		if len(solution_2[truck_added]) == 2:
			insertion_location = 1
		else:
			insertion_location = np.random.randint(1,len(solution_2[truck_added])-1)
		# print(solution_2[truck_added] )
		# print(insertion_location)


		solution_2[truck_added] = solution_2[truck_added][:insertion_location]+segment + solution_2[truck_added][insertion_location:]
		# print(solution_2[truck_added])


		all_ele = []
		for t in solution_2:
			all_ele+=t

		# print("LEN:",len(set(all_ele)))
		# if len(set(all_ele)) < 101: exit()
		# exit()
		# exit()
		# exit()
		return solution_2


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

	def iterate_on_solution(self,truck_paths,selector,iterations=1000,stop_if_no_progress=False):
			#For every vehicle, for every customer, for every location

			solution = truck_paths
			objective_value = self.calculate_total_distance(solution)
			print("initial:",objective_value)
			for step in range(iterations):
				previous_value = objective_value
				swaps = self.evaluate_neighbors(solution,sample_neighbors=True,sample_size=100)
				chosen = selector(swaps,previous_value,step,iterations)
				if chosen is not None:
					solution,_ = self.apply_swap(solution,chosen)
				objective_value = self.calculate_total_distance(solution)
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



