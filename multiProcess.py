import random
import numpy
import math
import pickle
import time
from multiprocessing import Pool
start_time = time.time()

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
        return self.score

    def make_action(self, action, price, i):
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

def update_wallet_i(wallet, price):
    res = wallet.update_score(price)
    return res

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

    def print_avg_score(self, epoch):
        avg = 0
        for wallet in self.list_wallet:
            avg = avg + wallet.score
        print(
            "Average score for this generation is :" + str(avg / len(self.list_wallet))
        )
        with open('avg.csv','a') as fd:
            fd.write(str(epoch) + "," + str(avg / len(self.list_wallet)) + '\n')

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
            W1, W2 = crossover(self.layers, father, mother)
            new_individuals.append(W1)
            new_individuals.append(W2)

        return new_individuals

    def create_next_generation(self, best_individuals):
        len_best = len(best_individuals)
        len_old = len(self.list_individual) - len_best

        self.list_individual.clear()
        self.list_wallet.clear()

        self.list_individual.append(best_individuals[0]) # Keep best of best

        # for i in range(len_best): # Copy best individual
        # 	self.list_individual.append(best_individuals[i])

        self.list_individual = self.list_individual + self.create_new_from_old_gen(
            best_individuals, len_best - 1
        )

        for i in range(len_old - 1):  # fill with new random ones
            self.list_individual.append(init_NN(self.layers))

        for i in range(len_best + len_old):  # reset wallets
            self.list_wallet.append(
                Wallet(self.initial_fees_rate, self.initial_money, self.initial_btc)
            )
        best_individuals.clear()

    def save_individuals(self):
        for i in range(len(self.list_individual)):
            l = str(self.layers)
            filename = "saves/l_" + l
            filename = filename + "_nb_" + str(i) + ".dat"
            filename = filename.replace(", ", "_")
            filename = filename.replace("[", "")
            filename = filename.replace("]", "")
            f = open(filename,'wb')
            pickle.dump(self.list_individual[i], f)
            f.close()

    def load_individuals(self):
        for i in range(len(self.list_individual)):
            l = str(self.layers)
            filename = "saves/l_" + l
            filename = filename + "_nb_" + str(i) + ".dat"
            filename = filename.replace(", ", "_")
            filename = filename.replace("[", "")
            filename = filename.replace("]", "")
            f = open(filename,'rb')
            example_dict = pickle.load(f)
            self.list_individual[i] = example_dict
            f.close()

    def mutate_all(self, freq, rate):
        for i in range(1, len(self.list_individual)): # Don't mutate best
            mutate( self.layers, self.list_individual[i], freq, rate)

    def edit_wallet(self, btc, money, last_action, i):
        self.list_wallet[i].btc = btc
        self.list_wallet[i].money = money
        self.list_wallet[i].last_action = last_action

def predict(population, layers, price, X, i):
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
    population.list_wallet[i].make_action(action, price, i)
    return action


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
    return random.uniform(father_w + 0.5, mother_w - 0.5)


def crossover(layers, father, mother):
    rng = random.uniform(0, 100)
    rate = random.uniform(0, 100)
    W1 = []
    W2 = []
    for l in range(1, len(layers)):
        if l == 1:
            W1.append([])
            W2.append([])
        W1.append([])
        W2.append([])
        for j in range(1, layers[l] + 1):
            if j == 1:
                W1[l].append([])
                W2[l].append([])
            W1[l].append([])
            W2[l].append([])
            for i in range(layers[l - 1] + 1):
                W1[l][j].append([])
                W2[l][j].append([])
                if rng >= rate:
                    W1[l][j][i] = father[l][j][i]
                    W2[l][j][i] = mother[l][j][i]
                else:
                    W1[l][j][i] = mother[l][j][i]
                    W2[l][j][i] = father[l][j][i]


    return W1, W2


def mutate(layers, W, freq, rate):
    for l in range(1, len(layers)):
        for j in range(1, layers[l] + 1):
            for i in range(layers[l - 1] + 1):
                rng = random.uniform(0, 100) / 100
                if (rng <= freq):
                    sign = random.uniform(-1, 1)
                    if sign >= 0:
                        W[l][j][i] = W[l][j][i] + (W[l][j][i] * rate)
                    else:
                        W[l][j][i] = W[l][j][i] - (W[l][j][i] * rate)


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


def predict_individual(population, i, filename):
    for line in get_all_line_csv(filename):
        X = get_X(line)
        X.pop() # rm timestamp
        price = X[3]
        maxi = 19891.99
        mini = 0.06
        price = price * (maxi - mini) + mini
        predict(population, population.layers, price, X, i)
        #print(population.list_wallet[i].money)
    return update_wallet_i(population.list_wallet[i], price)

def predict_individual_log(population, i, filename):
    for line in get_all_line_csv(filename):
        X = get_X(line)
        X.pop() # rm timestamp
        price = X[3]
        maxi = 19891.99
        mini = 0.06
        price = price * (maxi - mini) + mini
        action = predict(population, population.layers, price, X, i)
        with open("log_actions_" + str(i) + ".csv",'a') as fd:
            fd.write(str(price) + "," + str(action) + '\n')
    print(population.list_wallet[i].money)


if __name__ == "__main__":
    #filename = "coinbaseUSD_1min_clean.csv"
    # filename = "coinbaseUSD_1M.csv"
    filename = "foo2.csv"
    layers = [7, 20, 10, 5, 3]
    epochs = 400
    starting_balance = 100
    keep_best = 3
    nb_population = 10  
    btc = 0
    fees_rate = 0.5
    population = Population(nb_population, layers, fees_rate, starting_balance, btc)
    for epoch in range(epochs):
        population.reset_all_scores(starting_balance)
        p = Pool()
        params = []
        for i in range(len(population.list_individual)):
            params.append((population, i, filename))
        result = p.starmap(predict_individual, params)
        p.close()
        p.join()   

        for i in range(len(population.list_individual)):
            population.list_wallet[i].score = result[i]

        population.print_scores()
        population.print_avg_score(epoch)
        population.save_individuals()
        best_individuals = population.select_best_individual(keep_best)
        if epoch < epochs - 1:
            population.create_next_generation(best_individuals)
            population.mutate_all(0.2, 0.03) # 20% chance of mutate neuron
        print("--- %s seconds ---" % (time.time() - start_time))

    # log actions of the best indivudual
    for i in range(len(population.list_individual)):
        predict_individual_log(population, i, filename)
    #population.print_scores()