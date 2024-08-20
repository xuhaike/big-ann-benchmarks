import os
import yaml


# dataset_name="msmarco-10M"
# dataset_name="wikipedia-35M"
dataset_name="openai-2M"

plot_name="819aft"

if dataset_name=="msmarco-10M":
    # yaml_path="data/msmarco_websearch_20clustered/msmarco-10M-20clustered-general-runbook.yaml"
    # yaml_path="data/msmarco_websearch_20clustered_original/msmarco-10M-20clustered-general-runbook.yaml"
    yaml_path="data/msmarco_websearch_20clustered_dirichlet/msmarco-10M-20clustered-general-runbook.yaml"
    max_points=10000000
elif dataset_name=="wikipedia-35M":
    # yaml_path="data/wikipedia_cohere_60clustered/wikipedia-35M-60clustered-general-runbook.yaml"
    # yaml_path="data/wikipedia_cohere_60clustered_original/wikipedia-35M-60clustered-general-runbook.yaml"
    yaml_path="data/wikipedia_cohere_60clustered_dirichlet/wikipedia-35M-60clustered-general-runbook.yaml"
    max_points=35000000
elif dataset_name=="openai-2M":
    # yaml_path="data/wikipedia_cohere_60clustered/wikipedia-35M-60clustered-general-runbook.yaml"
    # yaml_path="data/wikipedia_cohere_60clustered_original/wikipedia-35M-60clustered-general-runbook.yaml"
    # yaml_path="data/OpenAIArXiv_10clustered_dirichlet/openai-2M-10clustered-general-runbook.yaml"
    # yaml_path="data/OpenAIArXiv_5clustered_dirichlet/openai-2M-5clustered-general-runbook.yaml"
    yaml_path="data/OpenAIArXiv_10clustered_dirichlet/openai-2M-10clustered-general-runbook.yaml"
    max_points=2321096

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)


insert_time=[0]*max_points
delete_time=[0]*max_points

# Access data
print(data[dataset_name]['max_pts'])
sum_points=[0]*max_points
max_time=0
for key in data[dataset_name]:
    print(key,data[dataset_name][key])
    if isinstance(data[dataset_name][key], dict):
        max_time=max(max_time,int(key))
        if data[dataset_name][key]["operation"] == "insert":
            start=data[dataset_name][key]["start"]
            end=data[dataset_name][key]["end"]
            # for x in range(start,end):
                # insert_time[x]=int(key)
            sum_points[int(key)]+=end-start
        elif data[dataset_name][key]["operation"] == "delete":
            start=data[dataset_name][key]["start"]
            end=data[dataset_name][key]["end"]
            # for x in range(start,end):
                # delete_time[x]=int(key)
            sum_points[int(key)]-=end-start

stats_points=[]
num_points=0
for t in range(max_time):
    num_points+=sum_points[t]
    stats_points.append(num_points)
    print(t,num_points)
# for t in range(max_time):
#     for i in range(max_points):
#         if insert_time[i]==t:
#             num_points+=1
#         if delete_time[i]==t:
#             num_points-=1
#     stats_points.append(num_points)

import matplotlib.pyplot as plt

plt.plot(stats_points)

# plt.savefig('stats_points.pdf',format="pdf")
plt.savefig(f'stats_points_{dataset_name}_{plot_name}.png',dpi=400)