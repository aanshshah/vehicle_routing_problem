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
		self.name = "iterative_sol_evo_and_twoopt"
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

	def approach(self,pop_size=50,top_k=5):
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
		if step % 1000 == 0:  #Every 100 steps, apply two-opt
			ranked_pop = [(self.iterate_on_2optSwap(p,iterations=10,stop_if_no_progress=True,stochastic=False,p=0.5)[0],s) for p,s in ranked_pop]

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

	def recombine(self,solution_1,solution_2,reverse_p=0.1): #Should i use numpy arrays??? (This applies to the swap approach too)
		
		start = np.random.randint(1,len(solution_1)-1)
		if -1 in solution_1[start:]:
			max_end = start + solution_1[start:].index(-1)
		else:
			max_end = start + len(solution_1[start:])

		end = np.random.randint(start+1,max_end) if start+1 < max_end else max_end

	
		segment = solution_1[start:end].copy()

		if np.random.rand() < reverse_p:
			segment.reverse()

		solution_2 = copy.deepcopy(solution_2)

		for c in segment:
			# if c != -1:
			# print(c)
			solution_2.remove(c)

		insertion_location = np.random.randint(1,len(solution_2))
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

	def iterate_on_2optSwap(self,truck_paths,iterations=1,stop_if_no_progress=True,stochastic=False,p=0.5):
		#For every vehicle, for every customer, for every location

		solution = truck_paths
		objective_value = self.calculate_total_distance(solution)
		for step in range(iterations):
			previous_value = objective_value

			for i in range(1,len(solution)-1):
				for j in range(i+1,len(solution)-1):
					if stochastic and np.random.rand() < p:
						continue
					flipped = solution[i:j+1].copy()
					flipped.reverse()
					new_route = solution[0:i] + flipped+solution[j+1:]
					value = self.calculate_total_distance(new_route)
					if self.check_within_capacity(new_route):
						if value < objective_value:
							solution = new_route
							objective_value = value
					
			if stop_if_no_progress:
				if previous_value == objective_value: break;

		return solution,objective_value


	




