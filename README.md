# ReinforcementLearning_Crypto
Learning how to make a Reinforcement Learning algorithm

We create a population of X individuals with X wallets.
For start each wallets has 1000 "euros" inside it and 0 "BTC tokens".

We create a neural network with given layers. Each nodes are between [-1, 1].
A prediction method will get what action to make depending on the NN and the inputs.
Inputs are a line in a BTC history CSV file.

With the prediction we update the wallets of each individuals.

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
  - [X] Mutation
    - [X] Random propability to mutate neurons in the NN.
      - [ ] Dynamic mutation rate/probabilty depending epochs?
   - [ ] Profit?
