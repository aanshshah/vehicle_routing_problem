class Strategy:
	def __init__(self, instance):
		self.instance = instance
		self.parse_instance()
		self.attempts = 0
		self.name = 'Strategy Name'
		self.vehicle_map = {i : (self.vehicle_capacity, 0, [0]) for i in range(self.num_vehicles)}
		self.paths = []
		self.distance = 0
		
	def parse_instance(self):
		self.depo_coords, self.specs, self.customer_info = self.instance
		self.num_customers = self.specs[0]
		self.num_vehicles = self.specs[1]
		self.vehicle_capacity = self.specs[2]

	def approach(self):
		pass

	def run(self):
		solution = self.approach()
		if solution:
			self.paths, self.distance = solution
		else:
			self.paths, self.distance = None, None
		return self.paths, self.distance