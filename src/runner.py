import argparse
from random_solution import RandomStrategy
from evolutionary import Evolution_Solution
import os
from twoopt import TwoOpt_Solution
from combined_hill_2opt import Combined_Hill2OPT_TwoOpt
from evo_2opt_sim_anneal import Combined_Evo_TwoOpt
from evo_2opt_sim_anneal2 import Combined_Evo_TwoOpt as Combined_Evo_TwoOpt2
from evo_2opt_sim_anneal3 import Combined_Evo_TwoOpt as Combined_Evo_TwoOpt3
from multiple_process import run_processes
import time
import json
#import pickle

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


def format_path(paths, save_solution=False):
    formatted_path = '0 '
    for path in paths:
        formatted_path += ' '.join(str(x) for x in path)
        formatted_path += ' '
    return formatted_path

def get_test_files():
    test_files = []
    for file in os.listdir('input'):
        test_files.append(os.path.join('input', file))
    return test_files

def run_all():
    for file in get_test_files():
        run_single(file)

def run_single(file, print_out=True):
    output_string = ''
    instance = read_instance(file)
    with open('src/decision_boundary_2.json', 'r') as fp:
        decision_boundary = json.load(fp)
    parts = file.split('/')
    pred = parts[0] if len(parts) == 1 else parts[-1]
    #strategy = decision_boundary.get(pred, 'evo2optsa')
    #strategy = get_strategy(strategy, instance)
    start_time = time.time()
    results = run_processes([Combined_Evo_TwoOpt(instance), Combined_Evo_TwoOpt2(instance), Combined_Evo_TwoOpt3(instance)])
    end_time = time.time()
    #print(results, time.time() - start_time)
    best = None
    best_dist = float("inf")
    for paths, distance in results:
        if distance < best_dist:
            best_dist = distance
            best = (paths, distance)
    paths, distance = best
    #paths, distance = strategy.run()
    #end_time = time.time()
    elapsed = end_time - start_time
    if paths and distance and print_out:
        if file[-1] == '/':
            output_string += "Instance: " + str(file.split('/')[1])
        else:
            output_string += "Instance: " + str(file)
        output_string += " Time: " + str(round(elapsed, 2))
        output_string += " Result: " + str(round(distance, 2))
        output_string += " Solution: "+format_path(paths, save_solution=True)
        print(output_string)
    else:
        print("NO SOLUTION FOUND FOR {0}".format(file))

def get_strategy(strategy, instance, seed=0):
    if strategy == 'evo2optsa':return Combined_Evo_TwoOpt(instance)
    else:return Combined_Hill2OPT_TwoOpt(instance)
#    return Combined_Evo_TwoOpt(instance)
#    if strategy == '2opt':
#        return TwoOpt_Solution(instance)
#    elif strategy == 'evo':
#        return Evolution_Solution(instance)
#    elif strategy == 'evo2opt':
#        return Combined_Evo_TwoOpt(instance)
#    else:
#        return Combined_Hill2OPT_TwoOpt(instance) #Combined_Evo_TwoOpt(instance)

def main(filename):
    run_single(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?')
    args = parser.parse_args()
    main(args.filename)
