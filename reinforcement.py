import random
import numpy
import math

class Wallet:
	def __init__(self, fees_rate, money, btc):
		self.score = 0
		self.fees_rate = fees_rate
		self.money = money
		self.btc = btc
		self.last_action = "SELL"
	def update_score(self, price):
		self.score = self.score + self.money
		self.score = self.score + (self.btc * price)
	def make_action(self, action, price):	
		if action == "BUY" and self.last_action == "SELL" :
			#print(action)
			self.btc = self.btc + self.money / price
			self.money = self.money - self.money 
			self.last_action = "BUY"
		elif action == "SELL" and self.last_action == "BUY":
			#print(action)
			self.money = self.money + self.btc * price
			self.btc = self.btc - self.btc
			self.last_action = "SELL"
		elif action == "HOLD":
			#print(action)
			self.btc = self.btc
			self.money = self.money
		else:
			#print("Cannot " + action + ", HOLD instead")
			self.btc = self.btc
			self.money = self.money
		# print("My btc: " + str(self.btc))
		# print("My money: " + str(self.money))
		# print(" ")

class Population:
	def __init__(self, nb_population, layers, fees_rate, money, btc):
		self.list_individual = []
		self.list_wallet = []

		for i in range(nb_population):
			self.list_individual.append(init_NN(layers))
			self.list_wallet.append(Wallet(fees_rate, money, btc))
	def update_all_scores(self, price):
		for wallet in self.list_wallet:
			wallet.update_score(price)


def get_all_predictions(layers, W, Xinput):
	Xinput.append(1) # Bias
	X = []
	for l in range(len(layers)):
		if l == 0:
			X.append([])
			pos = 0
			for i in range(len(Xinput)):
				X[l].append([])
				X[l][i] = Xinput[pos]
				pos = pos + 1
		else:
			X.append([])
			for i in range(layers[l] + 1):
				X[l].append([])
				X[l][i] = 0

	for l in range(1, len(layers)):
		for j in range(1, layers[l] + 1):
			res = 0.0
			for i in range(layers[l - 1] + 1):
				res = res + W[l][j][i] * X[l - 1][i]

			X[l][j] = math.tanh(res)
	return  X[len(layers) - 1]

def init_NN(layers):
	W = []
	for l in range(1, len(layers)):
		if l == 1:
			W.append([])
		W.append([])
		for j in range(1, layers[l] + 1):
			if j == 1:
				W[l].append([])
			W[l].append([])
			for i in range(layers[l - 1] + 1):
				W[l][j].append([])
				W[l][j][i] = random.uniform(-1, 1)
	return W

def get_next_action(predictions):
	predictions.pop(0)
	action = predictions.index(max(predictions))
	if action == 0:
		return "BUY"
	if action == 1:
		return "HOLD"
	if action == 2:
		return "SELL"
			
def crossover_w(father_w, mother_w):
	return random.uniform(father_w, mother_w)

def crossover(layers, father, mother):
	W = []
	for l in range(1, len(layers)):
		if l == 1:
			W.append([])
		W.append([])
		for j in range(1, layers[l] + 1):
			if j == 1:
				W[l].append([])
			W[l].append([])
			for i in range(layers[l - 1] + 1):
				W[l][j].append([])
				W[l][j][i] = crossover_w(father[l][j][i], mother[l][j][i])
	return W

def predict(layers, population, price):
	for i in range(len(population.list_individual)):
				predictions = get_all_predictions(layers, population.list_individual[i], X)
				action = get_next_action(predictions)
				population.list_wallet[i].make_action(action, price)

def get_X(line):
	X = line.split(',')
	X = X[1:]
	X = [float(i) for i in X]
	return X

def get_all_line_csv(filename):
	f = open(filename,'r')
	lines = f.readlines()[1:]
	f.close()
	return lines

if __name__ == '__main__':
	filename = "coinbaseUSD_1W.csv"
	layers = [8, 5, 5, 3]
	epochs = 10
	population = Population(100, layers, 0.5, 1000, 0)

	for epoch in range(epochs):
		for line in get_all_line_csv(filename):
			X = get_X(line)
			price = X[3]
			predict(layers, population, price)

		population.update_all_scores(price)
