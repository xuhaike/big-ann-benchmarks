import argparse
import os
import numpy as np
import random
import yaml

from scipy.cluster.vq import vq, kmeans2
from typing import Tuple
from benchmark.datasets import DATASETS

def cluster_and_permute(
    data, num_clusters
) -> Tuple[np.ndarray[int], np.ndarray[int]]:
    """
    Cluster the data and return permutation of row indices
    that would group indices of the same cluster together
    """
    npts = np.shape(data)[0]
    sample_size = min(100000, npts)
    sample_indices = np.random.choice(range(npts), size=sample_size, replace=False)
    sampled_data = data[sample_indices, :]
    centroids, sample_labels = kmeans2(sampled_data, num_clusters, minit="++", iter=10)
    labels, dist = vq(data, centroids)

    count = np.zeros(num_clusters)
    for i in range(npts):
        count[labels[i]] += 1
    print("Cluster counts")
    print(count)

    offsets = np.zeros(num_clusters + 1, dtype=int)
    for i in range(0, num_clusters, 1):
        offsets[i + 1] = offsets[i] + count[i]

    permutation = np.zeros(npts, dtype=int)
    counters = np.zeros(num_clusters, dtype=int)
    for i in range(npts):
        label = labels[i]
        row = offsets[label] + counters[label]
        counters[label] += 1
        permutation[row] = i

    return offsets, permutation


def write_permuted_data(
        data,
        permutation:np.ndarray[int],
        output_data_file:str
):
    permuted_data = data[permutation,:]

    shape = np.shape(permuted_data)
    with open(output_data_file, 'wb') as df:
        df.write(shape[0].to_bytes(4, 'little'))
        df.write(shape[1].to_bytes(4, 'little'))
        df.write(permuted_data)


def create_runbook(
    dataset_str:str,
    offsets:np.ndarray[int],
    permutation:np.ndarray[int],
    num_clusters:int, 
    output_yaml_file:str
):

    if "msmarco" in dataset_str:
        random.seed(8160)
        rng = np.random.default_rng(8160)
        checkpoint_per_step=10
        blocks_per_cluster=100
    elif "wikipedia" in dataset_str:
        random.seed(8161)
        rng = np.random.default_rng(8161)
        checkpoint_per_step=10
        blocks_per_cluster=100
    elif "openai" in dataset_str:
        random.seed(8164)
        rng = np.random.default_rng(8164)
        checkpoint_per_step=3
        blocks_per_cluster=100

    # rng=np.random.default_rng()

    operation_list = []
    num_operations = 1
    active_points = 0
    max_pts = 0

    

    blocks=[]
    for c in range(num_clusters):
        current_block=[]
        start=offsets[c]
        end=offsets[c+1]
        block_size=(end-start)//blocks_per_cluster
        for i in range(blocks_per_cluster-1):
            current_block.append((start+i*block_size,start+(i+1)*block_size))
        current_block.append((start+(blocks_per_cluster-1)*block_size,end))
        blocks.append(current_block)

        # print(current_block)

    active_blocks=[[] for _ in range(num_clusters)]
    cur_block_cursor=[0]*num_clusters

    num_active_blocks=[0]*num_clusters

    # In the first 60% time steps: consecutively insert 10% points and then remove a random fraction of the index size between 30% to 60%, repeat this for 6 rounds.

    # num_rounds=8
    # num_block_per_round = rng.dirichlet((100,50,30,15,10,5,3,1), num_clusters)

    num_rounds=6
    num_block_per_round = rng.dirichlet((100,15,10,5,3,1), num_clusters)

    round_sum=[]
    for c in range(num_clusters):
        rng.shuffle(num_block_per_round[c])
        # num_block_per_round[c]=[int(x*100) for x in num_block_per_round[c]]
        # assert(sum(num_block_per_round[c])<=100)
    # for i in range(num_rounds):
        # round_sum.append((sum([num_block_per_round[c][i] for c in range(num_clusters)]),i))
    # round_sum=sorted(round_sum)
    # round_sum=round_sum[1:]+round_sum[:1]
    # for c in range(num_clusters):
        # num_block_per_round[c]=[num_block_per_round[c][y] for (x,y) in round_sum]

    num_block_per_round=(num_block_per_round*blocks_per_cluster).astype(int)

    for round in range(num_rounds-1):
        for c in range(num_clusters):
            for step in range(num_block_per_round[c][round]):
                #insertions
                block_id=cur_block_cursor[c]
                delta = blocks[c][block_id][1]-blocks[c][block_id][0]

                active_points += delta
                max_pts = max(max_pts, active_points)
                active_blocks[c].append(block_id)

                num_active_blocks[c]+=1

                # print('ins [', blocks[c][block_id][0], ', ', blocks[c][block_id][1], ')' , 'total:', active_points)

                # entry = [{'operation': 'insert'}, {'start': int(blocks[c][block_id][0])}, {'end': int(blocks[c][block_id][1])}]
                entry = {'operation': 'insert', 'start': int(blocks[c][block_id][0]), 'end': int(blocks[c][block_id][1])}
                operation_list.append((num_operations, entry))
                num_operations += 1

                cur_block_cursor[c]+=1
            
            # if num_operations%checkpoint_per_step==0:
            # operation_list.append((num_operations, [{'operation': str('search')}]))
            operation_list.append((num_operations, {'operation': str('search')}))
            num_operations += 1

                

        # delete a random fraction 30% to 60% of active points 
        
        print(round, num_operations, num_active_blocks[c],active_points)

        for c in range(num_clusters):

            if round<num_rounds-2:
                delete_steps=random.randint(int(num_active_blocks[c]*0.5),int(num_active_blocks[c]*0.9))
            else:
                # delete_steps=int(num_active_blocks[c]*0.2)
                delete_steps=0

            for step in range(delete_steps):        
                #deletions
                delete_type=random.randint(0,9)
                if delete_type==9 and len(active_blocks[c])>2:
                    delete_id=random.randint(1, len(active_blocks[c])-2)
                    block_id=active_blocks[c].pop(delete_id)
                elif delete_type==8:
                    block_id=active_blocks[c].pop()
                else:
                    block_id=active_blocks[c].pop(0)
                    
                delta = blocks[c][block_id][1]-blocks[c][block_id][0]
                
                num_active_blocks[c]-=1

                active_points -= delta
                # print('del [', blocks[c][block_id][0], ', ', blocks[c][block_id][1], ')' , 'total:', active_points)

                # entry = [{'operation': 'delete'}, {'start': int(blocks[c][block_id][0])}, {'end': int(blocks[c][block_id][1])}]
                entry = {'operation': 'delete', 'start': int(blocks[c][block_id][0]), 'end': int(blocks[c][block_id][1])}
                operation_list.append((num_operations, entry))
                num_operations += 1

            # if num_operations%checkpoint_per_step==0:
            # operation_list.append((num_operations, [{'operation': str('search')}]))
            operation_list.append((num_operations, {'operation': str('search')}))
            num_operations += 1


        print("round = ", round, "active points = ", active_points)

    # in the later 40% steps, insert and delete in a interleaving way so that the index size is stable
    
    for c in range(num_clusters):

        while cur_block_cursor[c]<len(blocks[c]):

            for step in range(1):
                #insertions
                block_id=cur_block_cursor[c]
                delta = blocks[c][block_id][1]-blocks[c][block_id][0]

                num_active_blocks[c]+=1

                active_points += delta
                max_pts = max(max_pts, active_points)
                active_blocks[c].append(block_id)

                print('ins [', blocks[c][block_id][0], ', ', blocks[c][block_id][1], 
                    ')' , 'total:', active_points)

                # entry = [{'operation': 'insert'}, {'start': int(blocks[c][block_id][0])}, {'end': int(blocks[c][block_id][1])}]
                entry = {'operation': 'insert', 'start': int(blocks[c][block_id][0]), 'end': int(blocks[c][block_id][1])}
                operation_list.append((num_operations, entry))
                num_operations += 1

                cur_block_cursor[c]+=1
                
                # operation_list.append((num_operations, {'operation': str('search')}))
                # num_operations += 1

            # if num_operations%checkpoint_per_step==0:
            #     # operation_list.append((num_operations, [{'operation': str('search')}]))
            #     operation_list.append((num_operations, {'operation': str('search')}))
            #     num_operations += 1

            for step in range(1):        
                #deletions
                delete_type=random.randint(0,9)
                if delete_type==9 and len(active_blocks[c])>2:
                    delete_id=random.randint(1, len(active_blocks[c])-2)
                    block_id=active_blocks[c].pop(delete_id)
                elif delete_type==8:
                    block_id=active_blocks[c].pop()
                else:
                    block_id=active_blocks[c].pop(0)

                delta = blocks[c][block_id][1]-blocks[c][block_id][0]
                
                num_active_blocks[c]-=1

                active_points -= delta
                print('del [', blocks[c][block_id][0], ', ', blocks[c][block_id][1], 
                    ')' , 'total:', active_points)
                # entry = [{'operation': 'delete'}, {'start': int(blocks[c][block_id][0])}, {'end': int(blocks[c][block_id][1])}]
                entry = {'operation': 'delete', 'start': int(blocks[c][block_id][0]), 'end': int(blocks[c][block_id][1])}
                operation_list.append((num_operations, entry))
                num_operations += 1
            
                # operation_list.append((num_operations, {'operation': 'search'}))
                # num_operations += 1

            # if num_operations%checkpoint_per_step==0:
            #     # operation_list.append((num_operations, [{'operation': str('search')}]))
            #     operation_list.append((num_operations, {'operation': str('search')}))
            #     num_operations += 1
            
            print("round = ", round, "active points = ", active_points)

        # if num_operations%checkpoint_per_step==0:
        # operation_list.append((num_operations, [{'operation': str('search')}]))
        operation_list.append((num_operations, {'operation': str('search')}))
        num_operations += 1

    with open(output_yaml_file, 'w') as yf:
        operation_list.sort(key = lambda x: x[0])
        sorted_dict = {}
        sorted_dict['max_pts'] = int(max_pts)
        for (k, v) in operation_list:
            sorted_dict[k]=v
        yaml_object = {}
        yaml_object[dataset_str] = sorted_dict
        yaml.dump(yaml_object, yf)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '-c', '--num_clusters',
        type=int,
        required=True
    )
    parser.add_argument(
        '-o', '--output_data_file',
        required=True
    )
    parser.add_argument(
        '-y', '--output_yaml_file',
        required=True
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    if ds.nb <= 10**7:
        data = ds.get_dataset()
    else:
        data = next(ds.get_dataset_iterator(bs=ds.nb))
    print(np.shape(data))

    offsets, permutation = cluster_and_permute(data, args.num_clusters)
    print(permutation)

    write_permuted_data(data=data, 
                         permutation=permutation,
                         output_data_file=args.output_data_file)

    # print("write permuted data completes")

    create_runbook(dataset_str=args.dataset,
                   offsets=offsets,
                   permutation=permutation, 
                   num_clusters=args.num_clusters,
                   output_yaml_file=args.output_yaml_file)


if __name__ == '__main__':
    main()
