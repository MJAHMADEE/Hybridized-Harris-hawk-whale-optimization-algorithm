import numpy as np
import genetic as ga
import ANN
import csv
import Neural_network as neural
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import SSA
import time
import math

def best(fitness):
    max_fitness = 0
    pos = -1
    for i in range(len(fitness)):
        if fitness[i] > max_fitness:
            max_fitness = fitness[i]
            pos = i
    return pos

def Calculate_mean(pop_weights_mat):
    mean = pop_weights_mat[0]
    for i in range(1, len(pop_weights_mat)):
        mean += pop_weights_mat[i]
    return mean / len(pop_weights_mat)

def load_data():
    data_inputs = []
    with open('/content/Hybridized-Harris-hawk-whale-optimization-algorithm/Data Sets/blood.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = np.array(row, dtype=float)
            data_inputs.append(row)
    data_outputs = []
    with open('/content/Hybridized-Harris-hawk-whale-optimization-algorithm/Data Sets/label_blood.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = np.array(row, dtype=float)
            data_outputs.append(row)
    return data_inputs, data_outputs

def update_WOA1(parent, agent, C, A):
    parent = np.concatenate([np.array(p).flatten() for p in parent])
    agent = np.concatenate([np.array(a).flatten() for a in agent])
    return parent - A * abs(C * agent - parent)

def update_WOA3(agent, parent, C, l):
    agent = np.concatenate([np.array(a).flatten() for a in agent])
    parent = np.concatenate([np.array(p).flatten() for p in parent])
    b = 1  # typically a constant in WOA
    D1 = abs(C * agent - parent)
    return D1 * np.exp(b * l) * np.cos(l * 2 * np.pi) + agent
# def flatten_and_concatenate(items):
#     flattened_items = []
#     for item in items:
#         if isinstance(item, dict):
#             # If item is a dictionary, extract its values and flatten them
#             values = list(item.values())
#             flattened_items.extend(np.array(values, dtype=float).flatten())
#         elif isinstance(item, (list, np.ndarray)):
#             # Flatten lists and arrays
#             flattened_items.extend(np.array(item, dtype=float).flatten())
#         else:
#             # Convert single elements to float and add them to the list
#             flattened_items.append(float(item))
#     return np.array(flattened_items, dtype=float)

# def update_WOA1(parent, agent, C, A):
#     parent = flatten_and_concatenate(parent)
#     agent = flatten_and_concatenate(agent)
#     return parent - float(A) * abs(float(C) * agent - parent)

# def update_WOA3(agent, parent, C, l):
#     agent = flatten_and_concatenate(agent)
#     parent = flatten_and_concatenate(parent)
#     b = 1  # typically a constant in WOA
#     D1 = abs(float(C) * agent - parent)
#     return D1 * np.exp(b * l) * np.cos(l * 2 * np.pi) + agent
data_inputs, data_outputs = load_data()
minmax = ANN.dataset_minmax(data_inputs)
ANN.normalize_dataset(data_inputs, minmax)

data_inputs = np.array(data_inputs)
print(data_inputs.shape)
data_outputs = np.array(data_outputs)
data_inputs, X, data_outputs, y = train_test_split(data_inputs, data_outputs, test_size=0.2, random_state=1)

sol_per_pop = 12
num_parents_mating = 6
num_generations = 200
mutation_percent = 20

HL1_neurons = data_inputs.shape[1] * 5
HL2_neurons = int(data_inputs.shape[1])
output_neurons = 2

weight_range_1 = -1
weight_range_2 = 1

initial_pop_weights = []
for curr_sol in np.arange(0, sol_per_pop):
    input_HL1_weights = np.random.uniform(low=weight_range_1, high=weight_range_2, size=(data_inputs.shape[1], HL1_neurons))
    HL1_HL2_weights = np.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL1_neurons, HL2_neurons))
    HL2_output_weights = np.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL2_neurons, output_neurons))
    initial_pop_weights.append([input_HL1_weights, HL1_HL2_weights, HL2_output_weights])

pop_weights_mat = np.array(initial_pop_weights, dtype=object)
pop_weights_vector = np.array([np.concatenate([w.flatten() for w in weights]) for weights in pop_weights_mat])

mean_finess = []
best_outputs = []
accuracies = {}
F = pop_weights_mat[0]
F_acc = 0

for generation in range(100):
    print("Generation : ", generation)
    fitness = ANN.fitness(pop_weights_mat, data_inputs, data_outputs, activation="tanh")
    fitness = np.array(fitness)
    mean_finess.append(Calculate_mean(fitness))
    pop_weights_mat = np.array(pop_weights_mat)
    indices = fitness.argsort()
    pop_weights_mat = pop_weights_mat[indices]
    pop_weights_mat = pop_weights_mat[::-1]
    fitness = np.sort(fitness)
    fitness = fitness[::-1]
    F = pop_weights_mat[0]
    c1 = 2 * math.exp(-(pow((4 * (generation + 1) / num_generations), 2)))
    pop_weights_mat[0] = SSA.update_leader(F, weight_range_1, weight_range_2, c1)
    pop_weights_mat = SSA.update_follower(pop_weights_mat, fitness, data_inputs, data_outputs)
    accuracies[generation] = fitness
    print("Fitness")
    print(fitness)

for generation in range(100):
    print("Generation : ", generation)
    pop_weights_vector = np.array([np.concatenate([w.flatten() for w in weights]) for weights in pop_weights_mat])

    fitness = ANN.fitness(pop_weights_mat, data_inputs, data_outputs, activation="sigmoid")
    mean_finess.append(Calculate_mean(fitness))

    print("Fitness")
    print(fitness)

    parents = ga.select_mating_pool(pop_weights_vector, fitness.copy(), num_parents_mating)

    offspring_crossover = ga.crossover(parents, offspring_size=(pop_weights_vector.shape[0] - parents.shape[0], pop_weights_vector.shape[1]))

    offspring_mutation = ga.mutation(offspring_crossover, mutation_percent=mutation_percent)

    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

plt.plot(mean_finess)
plt.show()

# Custom vector_to_mat function to match expected shapes
def custom_vector_to_mat(vector, weight_shapes):
    mat = []
    start = 0
    for shape in weight_shapes:
        size = np.prod(shape)
        mat.append(vector[start:start + size].reshape(shape))
        start += size
    return mat

pop_weights_mat = [custom_vector_to_mat(vector, [w.shape for w in pop_weights_mat[0]]) for vector in pop_weights_vector]
best_weights = pop_weights_mat[0]

acc, predictions = ANN.predict_outputs(best_weights, X, y, activation="sigmoid")
print(acc)
print(predictions)
print(y)
