# python general_runbook_gen.py --dataset msmarco-10M --num_clusters 100 --output_data_file data/msmarco_websearch_100clustered/vectors.bin.crop_nb_10000000 --output_yaml_file data/msmarco_websearch_100clustered/msmarco-10M-100clustered-general-runbook.yaml

# python general_runbook_gen.py --dataset wikipedia-35M --num_clusters 100 --output_data_file data/wikipedia_cohere_100clustered/wikipedia_base.bin --output_yaml_file data/wikipedia_cohere_100clustered/wikipedia-35M-100clustered-general-runbook.yaml



# data_path="data/msmarco_websearch_20clustered_dirichlet_original"
# mkdir -p ${data_path}

# python general_runbook_gen.py --dataset msmarco-10M --num_clusters 20 --output_data_file ${data_path}/vectors.bin.crop_nb_10000000 --output_yaml_file ${data_path}/msmarco-10M-20clustered-general-runbook.yaml

# data_path="data/wikipedia_cohere_60clustered_dirichlet_original"
# mkdir -p ${data_path}

# python general_runbook_gen.py --dataset wikipedia-35M --num_clusters 60 --output_data_file ${data_path}/wikipedia_base.bin --output_yaml_file ${data_path}/wikipedia-35M-60clustered-general-runbook.yaml

# data_path="data/OpenAIArXiv_10clustered_dirichlet"
# mkdir -p ${data_path}

# python general_runbook_gen.py --dataset openai-2M --num_clusters 10 --output_data_file ${data_path}/openai_base.bin --output_yaml_file ${data_path}/openai-2M-10clustered-general-runbook.yaml

# data_path="data/OpenAIArXiv_10clustered_dirichlet"
# mkdir -p ${data_path}

# python general_runbook_gen.py --dataset openai-2M --num_clusters 10 --output_data_file ${data_path}/openai_base.bin --output_yaml_file ${data_path}/openai-2M-10clustered-general-runbook.yaml

data_path="data/msturing_20clustered_dirichlet"
mkdir -p ${data_path}

python general_runbook_gen.py --dataset msturing-10M --num_clusters 20 --output_data_file ${data_path}/base1b.fbin.crop_nb_10000000 --output_yaml_file ${data_path}/msturing-10M-20clustered-general-runbook.yaml