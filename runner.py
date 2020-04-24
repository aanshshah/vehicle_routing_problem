import argparse
from random_solution import RandomStrategy
from greedy_hill_climbing import GreedyHillClimbing
import os

def read_instance(filename):
		with open(filename, 'r') as f:
			lines = f.readlines()
		num_customers, num_vehicles, vehicle_capacity = [int(x) for x in lines[0].split()]
		_, depo_x, depo_y = [float(x) for x in lines[1].split()]
		customer_info = []
		idx = 1
		for line in lines[2:]:
			line = line.split()
			if line:
				customer = [idx] + [float(x) for x in line]
				idx += 1
				if customer:
					customer_info.append(customer) # idx, customer_demand, customer_x, customer_y
		return [depo_x, depo_y], [num_customers, num_vehicles, vehicle_capacity], customer_info


def format_path(paths, save_solution=False):
	formatted_path = ''
	if save_solution:
		for path in paths:
			formatted_path += ' '.join(str(x) for x in path)
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

def run_all(strategy):
	for file in get_test_files():
		run_single(file)

def run_single(file, strategy, print_out=True):
	instance = read_instance(file)
	strategy = get_strategy(strategy, instance)
	paths, distance = strategy.run()
	if paths and distance and print_out:
		print('Strategy: '+strategy.name)
		print(format_path(paths))
		print("distance: {0}".format(str(distance)))
	else:
		print("NO SOLUTION FOUND FOR {0}".format(file))


#find best seed across all files
def test_seeds(strategy_name):
	num_seeds = 10000
	best_seed = -1.0
	total_distance = 0
	min_distance = float("inf")
	for seed in range(num_seeds):
		for file in get_test_files():
			instance = read_instance(file)
			strategy = get_strategy(strategy_name, instance, seed)
			paths, distance = strategy.run()
			total_distance += distance
		if total_distance < min_distance:
			best_seed = seed
			min_distance = distance
	print(best_seed)
	return best_seed, min_distance

def get_strategy(strategy, instance, seed=0):
	if strategy == 'random':
		return RandomStrategy(instance, seed=seed)
	if strategy == 'random_greedy':
		return RandomStrategy(instance, greedy=True, seed=seed)
	if strategy == 'greedy_hill':
		return GreedyHillClimbing(instance)

def main(filename, single, strategy, test):
	if test and strategy:
		#best seed found for random was 0
		#random_greedy took really long and best seed was 
		test_seeds(strategy)
	elif single and filename:
		run_single(filename, strategy)
	else:
		run_all(strategy)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', nargs='?')
	parser.add_argument('-s', '--single', action='store_true')
	parser.add_argument('-a', '--strategy', default='random', choices=['random', 'random_greedy', 'greedy_hill'])
	parser.add_argument('-t', '--test', action='store_true')
	args = parser.parse_args()
	main(**vars(args))