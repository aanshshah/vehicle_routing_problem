from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import LeaveOneOut, cross_val_score 
import numpy as np

class StrategyInfo:
	def __init__(self, name, logname):
		self.logname = logname
		self.performance = {} #instance -> distance
		self.name = name
		self.read_log()

	@staticmethod
	def filter_name(instance_name):
		instance_name = instance_name.split('/')
		if len(instance_name) > 1:
			instance_name = instance_name[-1]
		else:
			instance_name = instance_name[0]
		return instance_name

	def read_log(self):
		with open(self.logname, 'r') as f:
			lines = f.readlines()
		for line in lines:
			line = line.split()
			instance_name = self.filter_name(line[1])
			distance = float(line[5])
			self.performance[instance_name] = distance

def get_instance_details(instance_name):
	with open('input/{0}'.format(instance_name), 'r') as f:
		lines = f.readlines()
	numCustomers, numVehicles, vehicleCapacity = [int(x) for x in lines[0].split()]
	return numCustomers, numVehicles, vehicleCapacity

def compare_performance(instance_one, instance_two):
	best_performances = {}
	assert len(instance_one.performance) == len(instance_two.performance)
	instances = list(instance_one.performance.keys())
	for instance in instances:
		dist_one = instance_one.performance[instance]
		dist_two = instance_two.performance[instance]
		if dist_one < dist_two:
			best_performances[instance] = (instance_one.name, dist_one)
		else:
			best_performances[instance] = (instance_two.name, dist_two)
	return best_performances

def generate_inputs(best_performance):
	#0 -> simulated_annealing
	#1 -> combined_hill
	X = []
	labels = []
	instance_names = []
	for instance_name, strat_info in best_performance.items():
		strategy_name, distance = strat_info
		label = 0 if strategy_name == 'simulated_annealing_2opt' else 1
		X.append(get_instance_details(instance_name))
		labels.append(label)
		instance_names.append(instance_name)
	return X, labels, instance_names

def pca(X): #62.5%
	model = PCA().fit(X)
	transformed = model.transform(X)
	return transformed

def KNN(X, y, instance_names): #93.5%
	neigh = KNeighborsClassifier(n_neighbors=2)
	neigh.fit(X, y)
	y_pred = neigh.predict(X)
	score, incorrect_names = accuracy(y_pred, y, instance_names)
	score = neigh.score(X, y)
	print(correct, incorrect_names)

def kmeans(X, y, instance_names): #62.5%
	model = KMeans(n_clusters=2, random_state=0).fit(X)
	correct, incorrect_names = accuracy(model.labels_, y, instance_names)
	print(correct, incorrect_names)

def accuracy(y_test, y_true, instance_names):
	correct = 0
	incorrect_names = []
	n = len(y_true)
	for i in range(n):
		if (y_test[i] >= 0.5 and y_true[i] == 1) or \
		(y_test[i] < 0.5 and y_true[i] == 0):
			correct += 1
		else:
			incorrect_names.append(instance_names[i])
	return (correct * 1.0)/n, incorrect_names

def linear_regression(X, labels, instance_names, cross_val=False): #93.5%
	# reg = LinearRegression()
	# loocv = LeaveOneOut()
	if cross_val:
		for i in range(2, 9):
			print("CV:", i)
			reg = LinearRegression()
			results_loocv = cross_val_score(reg, X, labels, cv=i)
			print(results_loocv)
			print("mean R^2:", results_loocv.mean())
	
	reg = LinearRegression().fit(X, labels)
	score = reg.score(X, labels)
	pred_label = reg.predict(X)
	correct, incorrect_names = accuracy(pred_label, labels, instance_names)
	return reg, correct, incorrect_names

def dbscan(X, y, instance_names): #62.5%
	best_score = -1
	best_labels = None
	best_eps = None
	best_ms = None
	best_incorrect = []
	eps_a = np.arange(0.01, 2.0, 0.01)
	for eps in eps_a:
		for min_sample in range(1, 3):
			clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(X)
			correct, incorrect_names = accuracy(clustering.labels_, y, instance_names)
			if correct > best_score:
				best_score = correct
				best_labels = clustering.labels_
				best_eps = eps
				best_ms = min_sample
				best_incorrect = incorrect_names
	print(best_score, best_incorrect)

def pretty_print(best_performance):
	for instance_name, strat_info in best_performance.items():
		print(instance_name, strat_info[0])

def validate_solution(logname):

	def get_route(routes):
		start = None
		end = None
		paths = []
		path = []
		for idx, customer in enumerate(routes):
			if customer == 0 and not start and not end:
				start = True
				path.append(customer)
			elif customer == 0 and not end and start:
				end = True
				path.append(customer)
			elif start and not end:
				path.append(customer)
			if start and end:
				start = None
				end = None
				paths.append(path)
				path = []
		return paths

	def check_num_vehicles(routes, num_vehicles):
		assert num_vehicles == len(routes), \
		"number of vehicles is {0} and routes length is {1}".format(num_vehicles, len(routes))

	def check_unique_passengers(routes, numCustomers):
		customers_visited = set()
		for route in routes:
			for customer in route:
				if customer == 0: 
					customers_visited.add(customer)
					continue

				assert customer not in customers_visited, \
				"customers are not unique: {0} visited again".format(customer)

				customers_visited.add(customer)
		# assert len(customers_visited) == numCustomers-1, "not all customers were visited"
		if len(customers_visited) != numCustomers:
			print(len(customers_visited), numCustomers)
			print("not all customers were visited")
		for i in range(numCustomers):
			if i not in customers_visited:
				print(i)
	def check_vehicle_starts_and_ends_at_depo(routes):
		for route in routes:
			assert route[0] == 0 and route[-1] == 0, "Vehicle must start and end at depo"

	def calculate_distance(x1, y1, x2, y2):
		return (((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))**0.5

	def calculate_total_distance(x, y, routes, demand, reported_distance):
		truck_x, truck_y = x[0], y[0]
		total_distance = 0.0
		for route in routes:
			for customer in route:
				customer_x, customer_y = x[customer], y[customer]
				total_distance += calculate_distance(truck_x, truck_y, customer_x, customer_y)
				truck_x, truck_y = customer_x, customer_y
		total_distance = round(total_distance, 2)
		if total_distance != reported_distance:
			print("Total distance is incorrect")
			print(total_distance, reported_distance)
		# assert reported_distance == total_distance, "Total distance is incorrect"

	def check_vehicle_capacity(routes, vehicle_capacity, demand):
		for route in routes:
			capacity = vehicle_capacity
			for customer in route:
				capacity -= demand[customer]
			assert capacity >= 0, "Max vehicle capacity was exceeded"

	def get_instance_information(instance_name):
		with open("input/{0}".format(instance_name), 'r') as f:
			lines = f.readlines()
		numCustomers, numVehicles, vehicleCapacity = [int(x) for x in lines[0].split()]
		demand = [] #the demand of each customer
		x = [] #the x coordinate of each customer
		y = [] #the y coordinate of each customer
		for idx, line in enumerate(lines[1:]):
			line = line.split()
			if line:
				demand.append(float(line[0]))
				x.append(float(line[1]))
				y.append(float(line[2]))
		return numCustomers, numVehicles, vehicleCapacity, demand, x, y


	instances = {}
	with open(logname, 'r') as f:
		lines = f.readlines()
	for line in lines:
		line = line.split()
		instance_name = line[1].split('/')
		if len(instance_name) > 1:
			instance_name = instance_name[-1]
		else:
			instance_name = instance_name[0]
		distance = line[5]
		path = [int(x) for x in line[8:]]
		instances[instance_name] = (distance, path)

	for instance_name, properties in instances.items():
		# if instance_name == '135_7_1.vrp':
		print(instance_name)
		distance = float(properties[0])
		routes = properties[1]
		numCustomers, numVehicles, vehicleCapacity, demand, x, y = get_instance_information(instance_name)
		routes = get_route(routes)
		check_num_vehicles(routes, numVehicles)
		check_unique_passengers(routes, numCustomers)
		check_vehicle_starts_and_ends_at_depo(routes)
		check_vehicle_capacity(routes, vehicleCapacity, demand)
		calculate_total_distance(x, y, routes, demand, distance)

def main():
	logs = ['results_attempt_pypy.log', 'results_combined_hill_2opt_pypy.log']
	names = ['simulated_annealing_2opt', 'combined_hill_2opt']
	strategy_one = StrategyInfo(names[0], logs[0])
	strategy_two = StrategyInfo(names[1], logs[1])
	best_performance = compare_performance(strategy_one, strategy_two)
	X, labels, instance_names = generate_inputs(best_performance)
	model, correct, incorrect_names = linear_regression(X, labels, instance_names)
	
	# print(model.coef_, model.intercept_)
	# pretty_print(best_performance)

if __name__ == '__main__':
	validate_solution("results_ref.log")
	# main()





