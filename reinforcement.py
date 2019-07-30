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
        self.score = self.money
        self.score = self.score + (self.btc * price)

    def make_action(self, action, price):
        if action == "BUY":
            # print(action)
            self.btc = self.btc + self.money / price
            self.money = self.money - self.money
            self.last_action = "BUY"
        elif action == "SELL":
            # print(action)
            self.money = self.money + self.btc * price
            self.btc = self.btc - self.btc
            self.last_action = "SELL"
        elif action == "HOLD":
            # print(action)
            self.btc = self.btc
            self.money = self.money
            self.last_action = "HOLD"
        else:
            print("\t\t\tERROR BAD ACTION!!!!")
        # print("My btc: " + str(self.btc))
        # print("My money: " + str(self.money))
        # print(" ")


class Population:
    def __init__(self, nb_population, layers, fees_rate, money, btc):
        self.layers = layers
        self.list_individual = []
        self.list_wallet = []
        self.initial_fees_rate = fees_rate
        self.initial_money = money
        self.initial_btc = btc

        for i in range(nb_population):
            self.list_individual.append(init_NN(self.layers))
            self.list_wallet.append(
                Wallet(self.initial_fees_rate, self.initial_money, self.initial_btc)
            )

    def update_all_scores(self, price):
        for wallet in self.list_wallet:
            wallet.update_score(price)

    def print_scores(self):
        i = 0
        for wallet in self.list_wallet:
            print("Wallet " + str(i) + " = " + str(wallet.score))
            i = i + 1

    def print_avg_score(self):
        avg = 0
        for wallet in self.list_wallet:
            avg = avg + wallet.score
        print(
            "Average score for this generation is :" + str(avg / len(self.list_wallet))
        )

    def reset_all_scores(self, money):
        for wallet in self.list_wallet:
            wallet.score = 0
            wallet.money = money
            wallet.btc = 0

    def select_best_individual(self, limit):
        if limit % 2 != 0:
            limit = limit + 1
        tmp_ind = self.list_individual.copy()
        tmp_wallet = self.list_wallet.copy()
        best_individuals = []
        for i in range(limit):
            max_score = 0
            index = 0
            for j in range(len(tmp_wallet)):
                if max_score < tmp_wallet[j].score:
                    index = j
                    max_score = tmp_wallet[index].score

            best_individuals.append(tmp_ind[index])
            tmp_ind.pop(index)
            tmp_wallet.pop(index)
        return best_individuals

    def create_new_from_old_gen(self, best_individuals, len_best):
        new_individuals = []

        for i in range(0, len_best, 2):
            father = best_individuals[i]
            mother = best_individuals[i + 1]
            W = crossover(self.layers, father, mother)
            new_individuals.append(W)

        random.shuffle(best_individuals)
        for i in range(0, len_best, 2):
            father = best_individuals[i]
            mother = best_individuals[i + 1]
            W = crossover(self.layers, father, mother)
            new_individuals.append(W)

        return new_individuals

    def create_next_generation(self, best_individuals):
        len_best = len(best_individuals)
        len_old = len(self.list_individual) - len_best

        self.list_individual.clear()
        self.list_wallet.clear()

        # self.list_individual.append(best_individuals[0]) # Keep best of best

        # for i in range(len_best): # Copy best individual
        # 	self.list_individual.append(best_individuals[i])

        self.list_individual = self.list_individual + self.create_new_from_old_gen(
            best_individuals, len_best
        )

        for i in range(len_old):  # fill with new random ones
            self.list_individual.append(init_NN(self.layers))

        for i in range(len_best + len_old):  # reset wallets
            self.list_wallet.append(
                Wallet(self.initial_fees_rate, self.initial_money, self.initial_btc)
            )
        best_individuals.clear()


def get_all_predictions(layers, W, Xinput):
    Xinput.append(1)  # Bias
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
    return X[len(layers) - 1]


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


def predict(layers, population, price, X):
    for i in range(len(population.list_individual)):
        if population.list_wallet[i].last_action == "SELL":
            X.append(-1)
        elif population.list_wallet[i].last_action == "BUY":
            X.append(1)
        elif population.list_wallet[i].last_action == "HOLD":
            X.append(0)
        else:
            X.append("This should make me crash")
        predictions = get_all_predictions(layers, population.list_individual[i], X)
        action = get_next_action(predictions)
        population.list_wallet[i].make_action(action, price)


def get_X(line):
    X = line.split(",")
    X = X[1:]
    X = [float(i) for i in X]
    return X


def get_all_line_csv(filename):
    f = open(filename, "r")
    lines = f.readlines()[1:]
    f.close()
    return lines


if __name__ == "__main__":
    filename = "coinbaseUSD_1D.csv"
    layers = [9, 3]
    epochs = 200
    starting_balance = 100
    keep_best = 2
    nb_population = 5
    btc = 0
    fees_rate = 0.5
    population = Population(nb_population, layers, fees_rate, starting_balance, btc)

    for epoch in range(epochs):
        population.reset_all_scores(starting_balance)
        for line in get_all_line_csv(filename):
            X = get_X(line)
            price = X[3]
            predict(layers, population, price, X)

        population.update_all_scores(price)
        population.print_scores()
        population.print_avg_score()
        best_individuals = population.select_best_individual(keep_best)
        if epoch < epochs - 1:
            population.create_next_generation(best_individuals)

    population.print_scores()

