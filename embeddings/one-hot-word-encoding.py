import numpy as np

samples = [
    'my dog ate my homework.',
    'the cat is on the table.',
    'the kitty is coming for food, and I like it.'
]

print(samples)

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

print(token_index)

max_length = 10
results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1


print(results)

