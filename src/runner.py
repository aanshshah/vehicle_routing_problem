import argparse
from random_solution import RandomStrategy
from evolutionary import Evolution_Solution
import os
from twoopt import TwoOpt_Solution
from combined_hill_2opt import Combined_Hill2OPT_TwoOpt
from evo_2opt_sim_anneal import Combined_Evo_TwoOpt
import time
import json
import pickle

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
    with open('src/decision_boundary.sav', 'rb') as fp:
        clf = pickle.load(fp)
    with open('src/label_to_name.json', 'r') as fp:
        label_to_name = json.load(fp)
    pred = str(clf.predict([instance[1]])[0])
    strategy = label_to_name[pred]
    strategy = get_strategy(strategy, instance)
    start_time = time.time()
    paths, distance = strategy.run()
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
        print(output_string)
    else:
        print("NO SOLUTION FOUND FOR {0}".format(file))

def get_strategy(strategy, instance, seed=0):
    if strategy == '2opt':
        return TwoOpt_Solution(instance)
    elif strategy == 'evo':
        return Evolution_Solution(instance)
    elif strategy == 'evo2opt':
        return Combined_Evo_TwoOpt(instance)
    else:
        return Combined_Hill2OPT_TwoOpt(instance) #Combined_Evo_TwoOpt(instance)

def main(filename):
    run_single(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?')
    args = parser.parse_args()
    main(args.filename)
