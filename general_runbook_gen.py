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

    operation_list = []
    num_operations = 1
    active_points = 0
    max_pts = 0


    blocks=[]
    for c in range(num_clusters):
        current_block=[]
        start=offsets[c]
        end=offsets[c+1]
        block_size=(end-start)//100
        for i in range(99):
            current_block.append((start+i*block_size,start+(i+1)*block_size))
        current_block.append((start+99*block_size,end))
        blocks.append(current_block)

        # print(current_block)

    active_blocks=[[] for _ in range(num_clusters)]
    cur_block_cursor=[0]*num_clusters

    for round in range(4):
        for c in range(num_clusters):
            for step in range(15):
                #insertions
                block_id=cur_block_cursor[c]
                delta = blocks[c][block_id][1]-blocks[c][block_id][0]

                active_points += delta
                max_pts = max(max_pts, active_points)
                active_blocks[c].append(block_id)

                print('ins [', blocks[c][block_id][0], ', ', blocks[c][block_id][1], 
                    ')' , 'total:', active_points)

                # entry = [{'operation': 'insert'}, {'start': int(blocks[c][block_id][0])}, {'end': int(blocks[c][block_id][1])}]
                entry = {'operation': 'insert', 'start': int(blocks[c][block_id][0]), 'end': int(blocks[c][block_id][1])}
                operation_list.append((num_operations, entry))
                num_operations += 1

                if num_operations%100==0:
                    # operation_list.append((num_operations, [{'operation': str('search')}]))
                    operation_list.append((num_operations, {'operation': str('search')}))
                    num_operations += 1

                cur_block_cursor[c]+=1

        if round<3:
            delete_steps=14
        else:
            delete_steps=3

        for c in range(num_clusters):
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
                
                active_points -= delta
                print('del [', blocks[c][block_id][0], ', ', blocks[c][block_id][1], 
                    ')' , 'total:', active_points)
                # entry = [{'operation': 'delete'}, {'start': int(blocks[c][block_id][0])}, {'end': int(blocks[c][block_id][1])}]
                entry = {'operation': 'delete', 'start': int(blocks[c][block_id][0]), 'end': int(blocks[c][block_id][1])}
                operation_list.append((num_operations, entry))
                num_operations += 1

                if num_operations%100==0:
                    # operation_list.append((num_operations, [{'operation': str('search')}]))
                    operation_list.append((num_operations, {'operation': str('search')}))
                    num_operations += 1


        print("round = ", round, "active points = ", active_points)

    
    for c in range(num_clusters):

        for round in range(40):

            for step in range(1):
                #insertions
                block_id=cur_block_cursor[c]
                delta = blocks[c][block_id][1]-blocks[c][block_id][0]

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

                if num_operations%100==0:
                    # operation_list.append((num_operations, [{'operation': str('search')}]))
                    operation_list.append((num_operations, {'operation': str('search')}))
                    num_operations += 1

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
                
                active_points -= delta
                print('del [', blocks[c][block_id][0], ', ', blocks[c][block_id][1], 
                    ')' , 'total:', active_points)
                # entry = [{'operation': 'delete'}, {'start': int(blocks[c][block_id][0])}, {'end': int(blocks[c][block_id][1])}]
                entry = {'operation': 'delete', 'start': int(blocks[c][block_id][0]), 'end': int(blocks[c][block_id][1])}
                operation_list.append((num_operations, entry))
                num_operations += 1
            
                # operation_list.append((num_operations, {'operation': 'search'}))
                # num_operations += 1

                if num_operations%100==0:
                    # operation_list.append((num_operations, [{'operation': str('search')}]))
                    operation_list.append((num_operations, {'operation': str('search')}))
                    num_operations += 1
            
            print("round = ", round, "active points = ", active_points)

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

    print("write permuted data completes")

    create_runbook(dataset_str=args.dataset,
                   offsets=offsets,
                   permutation=permutation, 
                   num_clusters=args.num_clusters,
                   output_yaml_file=args.output_yaml_file)


if __name__ == '__main__':
    main()
