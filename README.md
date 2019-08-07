# ReinforcementLearning_Crypto
Learning how to make a Reinforcement Learning algorithm

We create a population of X individuals with X wallets.

For start each wallets has 1 `USD` inside it and 0 `BTC tokens`.

We create a neural network with given layers. Each wieghts of the nodes are between [-1, 1].

A prediction method will get what action to make depending on the NN and the inputs.

It will update the score of the wallet (0.25% fees at each BUT/SELL action).

Inputs are a line in a BTC history CSV file. Mostly containing: `high`, `low`, `open`, `close`, `weight`, `volume`. Timestamps should never be feeded to the NN, else it will learn over the date instead of the patterns of candles.

The signal score is the money made at the end of the period in the CSV file.

## TODO
  
  - [ ] Crossover
    - [X] Selection: Get best individuals regarding score.
    - [X] Save best individuals (This can cause premature convergence over the long term. Because we can keep the best individual for ever)
    - [X] Fill rest of population:
      - [X] Fully new random
      - [X] Random propability to take Father's or Mother's neuron for child1 and child2
      - [X] ~~Average neurons~~ Seems to have a bad convergence property
      - [X] ~~Random between father and mother neurons~~ Seems to have a bad convergence property
      - [ ] Random crossover method at each epochs?
  - [X] Mutation
    - [X] Random propability to mutate neurons in the NN.
      - [X] Dynamic mutation rate/probabilty depending epochs? (Yes but poorly implemented)
      - [ ] Random mutation method for each individuals?
   - [ ] Profit?

## Branch noHistory
  This branch uses the `high`, `low`, `open`, `close`, `weight`, `volume` and `last_action` to predict the next action. This method seems to be quite effective, BUT I'm sure it won't be effective on new data.
  
  The problem with the real values is that what happends if we break the max value seen, the lower value seen? The data is only between 200$ (2015) up to 19000$ (2017) [2015-2019]. Thus, how would react the Neural Network with unbounded values?
  How should I generate them in real time? Would I have to retrain a model each time we hit a new max/min ? Would this be efficient?

  Even if we never hit new max/min, how would I efficiently append new data to my program? The standardization won't be anymore centered on 0. That could make all the learning useless.
  
  So good results for training data, less sure about real time values. I guess I overfit the curve. Maybe the I need to had some more history feeded to the NN.

## Branch Master
  This branch should not be the master branch, but the `Xhistory` branch. The `noHistory` branch should been a specific case of `Xhistory` with `X = 0`, but I can't manage to get the same results.
  
  Here we feed the NN with `X` lines of data from the CSV file. The goal here is to learn on patterns on multiples candle patterns. But I cannot find any good structure of the NN yet. The NN must be bigger, thus longer to train, longer to predict, longer to make an action, thus potential money loss.

## Branch ColorInput
  I wanted to  try with different inputs. Instead of working with normalize and standardize values, I wanted to work with the color of the candles.

  I can make an abstraction on value of the currency, playing with color of the candles could be a good way to generate safe revenue. I'm aware this is not optimized, but maybe it will be find a more general rule on when to BUY/SELL/HOLD.

  What I hope, is the NN to learn patterns with the history of candle colors.
  
  For example, if each candle is one day, what would be our reaction if we seen 6 red candles? Personnaly I would buy, since Crypto is a very speculative currency. And in my opinion, there will always be someone to try to make money over crashes.

  This reflection is **heavily biais** by my way of thinking of the market. In theory, it should learn it self the patterns with basic data like: `high`, `low`, `open`, `close`, `weight`, `volume`.
