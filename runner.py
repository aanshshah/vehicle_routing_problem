import argparse
from random_solution import RandomStrategy

def read_instance(filename):
		with open(filename, 'r') as f:
			lines = f.readlines()
		num_customers, num_vehicles, vehicle_capacity = [int(x) for x in lines[0].split()]
		_, depo_x, depo_y = [int(x) for x in lines[1].split()]
		customer_info = []
		idx = 1
		for line in lines[2:]:
			line = line.split()
			if line:
				customer = [idx] + [int(x) for x in line]
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

def main(instance):
	instance = read_instance(instance)
	random_strategy = RandomStrategy(instance)
	paths, distance = random_strategy.run()
	print(format_path(paths))
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args.filename)