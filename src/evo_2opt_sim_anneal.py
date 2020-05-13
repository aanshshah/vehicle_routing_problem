import random 
from utils import calculate_distance
from strategy import Strategy
# import numpy as np
import copy
import time
# from tqdm import tqdm
import math

class Combined_Evo_TwoOpt(Strategy):
    def __init__(self, instance, greedy=False, seed=0, epsilon=1):
        super().__init__(instance)
        random.seed(seed)
        self.greedy = greedy
        self.name = "iterative_sol_evo_and_twoopt"
        self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
        self.epsilon = epsilon
        self.start_time = time.time()
    
    def check_time(self):
        if time.time() - self.start_time > 290: return True
        else: return False    

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

    def approach(self,pop_size=100,top_k=10):
        population = [self.flatten(self.get_initial_solution(i)) for i in range(pop_size)]

        for step in range(100000):
            population = self.evolve(population,pop_size,top_k,step)
            if step % 100 == 0:
                self.paths = self.unflatten(population[0])
                self.distance = self.calculate_total_distance(population[0])

            if self.check_time(): break
        ranked_population = self.evaluate_fit(population, top_k)     
        self.paths = self.unflatten(ranked_population[0][0])
        self.distance = ranked_population[0][1] #self.calculate_total_distance(population[0])
        return self.paths, self.distance

    def evaluate_fit(self, population, top_k):     
        fitness = [(s,self.calculate_total_distance(s)+1000000*(not self.check_within_capacity(s))) for s in population]
        ranked_pop = sorted(fitness,key=lambda x: x[-1])[:top_k]
        return ranked_pop
    def evolve(self,population,pop_size,top_k,step,p=None):
        #fitness = [(s,self.calculate_total_distance(s)+1000000*(not self.check_within_capacity(s))) for s in population]

        #ranked_pop = sorted(fitness,key=lambda x: x[-1])[:top_k]
        ranked_pop = self.evaluate_fit(population,top_k)

        if step % 500 == 0:  #Every 100 steps, apply two-opt
            ranked_pop = [(self.iterate_on_2optSwap(p,iterations=2,stop_if_no_progress=False,t_0=6000/(step+1))[0],s) for p,s in ranked_pop]

        if self.check_time(): 
            return [x[0] for x in ranked_pop]

        new_pop = []
        for x in range(pop_size//(top_k*top_k)):
            for i in range(top_k):
                for j in range(top_k):
                    if i != j:
                        candidate = self.recombine(ranked_pop[i][0],ranked_pop[j][0])
                    else:
                        candidate = ranked_pop[i][0]
                    if not self.check_within_capacity(candidate):
                        candidate = ranked_pop[i][0]

                    new_pop.append(candidate)

        return new_pop

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
            solution_2.remove(c)

        insertion_location = random.randint(1,len(solution_2))
        solution_2= solution_2[:insertion_location]+segment + solution_2[insertion_location:]
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

            if total_d > self.vehicle_capacity:
                return False

        return True

    def calculate_total_distance(self,solution_flat):
        total_distance_traveled = 0

        _,_,start_x,start_y = self.customer_info[0]

        for l in solution_flat+[-1]:
            if l == -1:
                _,_,end_x,end_y = self.customer_info[0]
            else:
                _,_,end_x,end_y = self.customer_info[l]

            total_distance_traveled += calculate_distance(start_x, start_y, end_x, end_y)

            start_x = end_x
            start_y = end_y

        return total_distance_traveled

    def iterate_on_2optSwap(self,truck_paths,iterations=1,stop_if_no_progress=False,t_0=10):
        #For every vehicle, for every customer, for every location

        solution = truck_paths
        objective_value = self.calculate_total_distance(solution)
        c_it = 1
        a = 0.95
        #t_0 = 10 #1000
        t_f = 0.1
        #print(t_0)
        t = t_0
        beta = 1.0
        cur_iterations = 0
        while t >= t_f:
            #   print(objective_value)
            # print(t)
            if cur_iterations > iterations:
                break
            cur_iterations+=1
            for step in range(c_it):
                previous_value = objective_value

                for i in range(1,len(solution)-1):
                    for j in range(i+1,len(solution)-1):
                        flipped = solution[i:j+1].copy()
                        flipped.reverse()
                        new_route = solution[0:i] + flipped+solution[j+1:]
                        value = self.calculate_total_distance(new_route)
                        if not self.check_within_capacity(new_route):
                            continue

                        if value < objective_value:
                            p = 1
                        else:
                            delta = value - objective_value
                            p = math.exp(-1*delta/t)

                        r = random.random()

                        if r < p:
                            solution = new_route
                            objective_value = value

                        if self.check_time(): 
                        # self.paths = solution
                      #  self.distance = objective_value
                          return [solution, objective_value]    
            t = a * t
            c_it = int(beta*c_it)
            if stop_if_no_progress:
                if previous_value <= objective_value: break;
        #self.paths = solution
        #self.distance = objective_value
        return solution,objective_value

