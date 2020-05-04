import argparse, copy, math, random, time

class Instance:
	def __init__(self, filename):
		self.filename = filename
		self.read_instance_file()
		self.routes = [[] for _ in range(self.numVehicles) ]

	def read_instance_file(self):
		with open(self.filename, 'r') as f:
			lines = f.readlines()
		self.numCustomers, self.numVehicles, self.vehicleCapacity = [int(x) for x in lines[0].split()]
		self.demand = [] #the demand of each customer
		self.x = [] #the x coordinate of each customer
		self.y = [] #the y coordinate of each customer
		for idx, line in enumerate(lines[1:]):
			line = line.split()
			if line:
				self.demand.append(float(line[0]))
				self.x.append(float(line[1]))
				self.y.append(float(line[2]))

	def objective(self, loc_routes):
		dist = 0
		for i in range(self.numVehicles):
			tempRoute = loc_routes[i]
			if len(tempRoute) < 1: continue
			tempCust = tempRoute[0]
			dist += self.distance(self.x[0], self.y[0], self.x[tempCust], self.y[tempCust])

			for j in range(1, len(tempRoute)):
				currentCust = tempRoute[j]
				dist += self.distance(self.x[currentCust], self.y[currentCust], self.x[tempCust], self.y[tempCust])
				tempCust = currentCust
			dist += self.distance(self.x[0], self.y[0], self.x[tempCust], self.y[tempCust])
		return dist

	@staticmethod
	def distance(x1, y1, x2, y2):
		return (((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))**0.5

	def availableCapacity(self, vehicle):
		capacity = self.vehicleCapacity
		tempRoute = self.routes[vehicle]
		for i in range(len(tempRoute)):
			capacity -= self.demand[tempRoute[i]]
		return capacity

	def randomSol(self):
		for c in range(1, self.numCustomers):
			for t in range(int(1.5*self.numVehicles)):
				r = int(math.floor(random.random()*self.numVehicles))
				if self.availableCapacity(r) >= self.demand[c]:
					self.routes[r].append(c)
					break

	def twoopt(self):
		while True:
			v1 = int(math.floor(random.random()*self.numVehicles))
			v2 = int(math.floor(random.random()*self.numVehicles))
			if v1==v2 or len(self.routes[v1])<1 or len(self.routes[v2])<1: continue
			i1 = int(math.floor(random.random()*len(self.routes[v1])))
			i2 = int(math.floor(random.random()*len(self.routes[v2])))
			c1 = self.routes[v1][i1]
			c2 = self.routes[v2][i2]
			if (self.availableCapacity(v1) + self.demand[c1] < self.demand[c2]) or (self.availableCapacity(v2) + self.demand[c2] < self.demand[c1]):
				continue

			self.routes[v1][i1] = c2
			self.routes[v2][i2] = c1
			break

	def insertion(self):
		while True:
			v1 = int(math.floor(random.random()*self.numVehicles))
			v2 = int(math.floor(random.random()*self.numVehicles))
			if v1==v2 or len(self.routes[v1])<1: continue

			i1 = int(math.floor(random.random()*len(self.routes[v1])))
			i2 = int(math.floor(random.random()*len(self.routes[v2])))
			c1 = self.routes[v1][i1]
			

			if self.availableCapacity(v2) < self.demand[c1]: continue
			
			self.routes[v1].pop(i1)
			self.routes[v2].insert(i2, c1)
			
			break

	@staticmethod
	def routeClone(loc_routes):
		return copy.deepcopy(loc_routes)


def solve_instance(instance_filename, start_time):
	c_it = 5000
	a = 0.95
	t_0 = 1000
	t_f = 0.1
	t = t_0
	beta = 1.0

	instance = Instance(instance_filename)
	instance.randomSol()

	x_route = instance.routeClone(instance.routes)
	best_route = instance.routeClone(instance.routes)

	obj_x = instance.objective(instance.routes)
	obj_best = obj_x
	obj_cur = obj_x

	while t >= t_f:
		for i in range(c_it):
			if time.time() - start_time > 290: return obj_best, best_route
			r = random.random()
			if r <= 0.5:
				instance.twoopt()
			else:
				instance.insertion()
			obj_cur = instance.objective(instance.routes)
			if obj_cur < obj_x:
				p = 1
			else:
				delta = obj_cur - obj_x
				p = math.exp(-1*delta/t)

			r = random.random()

			if r < p:
				x_route = instance.routeClone(instance.routes)
				obj_x = obj_cur
			else:
				instance.routes = instance.routeClone(x_route)
			if obj_cur < obj_best:
				best_route = instance.routeClone(instance.routes)
				obj_best = obj_cur

		t = a * t
		c_it = int(beta*c_it)

	return obj_best, best_route

def format_solution(filename, distance, time, routes):
	output = ''
	formatted_paths = '0 '
	num_vehicles = len(routes)
	for i in range(num_vehicles):
		routes[i] = [0] + routes[i] + [0]
	for route in routes:
		formatted_paths += ' '.join(str(x) for x in route)
		formatted_paths += ' '
	instance_name = filename.split('/')
	if len(instance_name) > 1:
		instance_name = instance_name[1]
	else:
		instance_name = instance_name[0]
	output += "Instance: {0} Time: {1} Result: {2} ".format(instance_name, str(round(time, 2)), str(round(distance, 2)))
	output += "Solution: {0}".format(formatted_paths)
	return output

def main(filename, start_time=time.time()):
	distance, best_route = solve_instance(filename, start_time)
	elapsed_time = time.time() - start_time
	output = format_solution(filename, distance, elapsed_time, best_route)
	print(output)

if __name__ == '__main__':
	random.seed(0)
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	start_time = time.time()
	main(args.filename, start_time)
