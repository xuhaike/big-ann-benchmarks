# python general_runbook_gen.py --dataset msmarco-10M --num_clusters 100 --output_data_file data/msmarco_websearch_100clustered/vectors.bin.crop_nb_10000000 --output_yaml_file data/msmarco_websearch_100clustered/msmarco-10M-100clustered-general-runbook.yaml

# python general_runbook_gen.py --dataset wikipedia-35M --num_clusters 100 --output_data_file data/wikipedia_cohere_100clustered/wikipedia_base.bin --output_yaml_file data/wikipedia_cohere_100clustered/wikipedia-35M-100clustered-general-runbook.yaml

python general_runbook_gen.py --dataset msmarco-10M --num_clusters 20 --output_data_file data/msmarco_websearch_20clustered/vectors.bin.crop_nb_10000000 --output_yaml_file data/msmarco_websearch_20clustered/msmarco-10M-20clustered-general-runbook.yaml

python general_runbook_gen.py --dataset wikipedia-35M --num_clusters 60 --output_data_file data/wikipedia_cohere_60clustered/wikipedia_base.bin --output_yaml_file data/wikipedia_cohere_60clustered/wikipedia-35M-60clustered-general-runbook.yaml
