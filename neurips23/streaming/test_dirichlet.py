import argparse
import os
import numpy as np
import random
import yaml


# sample = np.random.default_rng().dirichlet((100,15,10,5,3), 20)

# print(sample)

# print((sample*100).astype(int))

num_clusters=20
# num_clusters=60

# num_rounds=8
# num_block_per_round = np.random.default_rng().dirichlet((100,50,30,15,10,5,3,1), num_clusters)

num_rounds=6
num_block_per_round = np.random.default_rng().dirichlet((100,15,10,5,3,1), num_clusters)

round_sum=[]
for c in range(num_clusters):
    np.random.default_rng().shuffle(num_block_per_round[c])
    # num_block_per_round[c]=[int(x*100) for x in num_block_per_round[c]]
    # assert(sum(num_block_per_round[c])<=100)
for i in range(num_rounds):
    round_sum.append((sum([num_block_per_round[c][i] for c in range(num_clusters)]),i))
# round_sum=sorted(round_sum)
# round_sum=round_sum[1:]+round_sum[:1]
for c in range(num_clusters):
    num_block_per_round[c]=[num_block_per_round[c][y] for (x,y) in round_sum]

num_block_per_round=(num_block_per_round*100).astype(int)

print(round_sum)

print(num_block_per_round)