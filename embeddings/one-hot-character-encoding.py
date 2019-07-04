import string

import numpy as np

samples = [
    'my dog ate my homework.',
    'the cat is on the table.',
    'the kitty is coming for food, and I like it.'
]

characters = string.printable

token_index = dict(zip(range(1, len(characters) + 1), characters))
max_length = 50

results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1


print(results)