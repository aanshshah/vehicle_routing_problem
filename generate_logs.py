import argparse
from random_solution import RandomStrategy
# from greedy_hill_climbing import GreedyHillClimbing
from stochastic_hill_climb import Iterative_Solution
from evolutionary import Evolution_Solution
import os
# from combine_approaches import Combined_Evo_TwoOpt
from twoopt import TwoOpt_Solution
from combined_hill_2opt import Combined_Hill2OPT_TwoOpt
from evo_2opt_sim_anneal import Combined_Evo_TwoOpt
import time
import attempt


def read_instance(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_customers, num_vehicles, vehicle_capacity = [int(x) for x in lines[0].split()]
        _, depo_x, depo_y = [float(x) for x in lines[1].split()]
        customer_info = [[0,0,depo_x, depo_y]]
        idx = 1
        for line in lines[2:]:
            line = line.split()
            if line:
                customer = [idx] + [float(x) for x in line]
                idx += 1
                if customer:
                    customer_info.append(customer) # idx, customer_demand, customer_x, customer_y
        # print(customer_info)
        return [depo_x, depo_y], [num_customers, num_vehicles, vehicle_capacity], sorted(customer_info,key=lambda x: x[0])


def format_path(paths, save_solution=True):
    formatted_path = '0 '
    if save_solution:
        for path in paths:
            formatted_path += ' '.join(str(x) for x in path)
            formatted_path += ' '
    else:
        for idx, path in enumerate(paths):
            formatted_path += 'truck {0}: '.format(str(idx + 1))
            formatted_path += ' '.join(str(x) for x in path)
            formatted_path += '\n'
    return formatted_path

def get_test_files():
    test_files = []
    for file in os.listdir('input'):
        test_files.append(os.path.join('input', file))
    return test_files

def run_all():
    strategies = ['random', '2opt', 'evo', 'evo_2opt', 'evo_2opt_sa']
    for strategy in strategies:
        log_name = 'results_{0}_final.log'.format(strategy)
        output = ''
        for file in get_test_files():
            print(strategy, file)
            output += run_single(file,strategy,log_name)
            output += '\n'
        with open('logs/{0}'.format(log_name), 'w') as f:
            f.write(output)

def run_single(file, strategy, log_name, print_out=True):
    output_string = ''
    instance = read_instance(file)
    strategy = get_strategy(strategy, instance)
    start_time = time.time()
    paths, distance = strategy.run()
    try:
        if strategy.distance < distance and strategy.paths:
            paths = strategy.paths
    except:
        pass
    end_time = time.time()
    elapsed = end_time - start_time
    if paths and distance and print_out:
        if file[-1] == '/':
            output_string += "Instance: " + str(file.split('/')[1])
        else:
            output_string += "Instance: " + str(file)
        output_string += " Time: " + str(round(elapsed, 2))
        output_string += " Result: " + str(round(distance, 2))
        output_string += " Solution: "+format_path(paths, save_solution=True)
        return output_string
    else:
        return "NO SOLUTION FOUND FOR {0}".format(file)

def get_strategy(strategy, instance, seed=0):
    if strategy == 'random':
        return RandomStrategy(instance, seed=seed)
    elif strategy == '2opt':
        return TwoOpt_Solution(instance)
    elif strategy == 'evo':
        return Evolution_Solution(instance)
    elif strategy == 'evo_2opt':
        return Combined_Evo_TwoOpt(instance)
    else: 
        return Combined_Hill2OPT_TwoOpt(instance) #Combined_Evo_TwoOpt(instance)

def main():
    run_all()

if __name__ == '__main__':
    main()
