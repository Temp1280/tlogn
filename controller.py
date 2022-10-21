# -*- coding: utf-8 -*-


import os



def run_TLoGN():
    max_edges = 40
    sample_nodes = 600
    sample_method = 3
    sample_ratio = 0.5
    score_method = 'att'
    loss = 'bce'
    use_gcn = 0
    time_score = 0

    gpu = 6

    temp_cmd = f'python run_model.py --max_nodes {max_edges} --sample_nodes {sample_nodes}' \
               f' --sample_method {sample_method} --sample_ratio {sample_ratio}' \
               f' --score_method {score_method} --loss {loss} --use_gcn {use_gcn}' \
               f' --time_score {time_score} --gpu {gpu}'
    os.system(temp_cmd)

run_TLoGN()



# def view_results():




def run_gcn():
    # gcn_layers_list = [1,2]
    # dataset_list = ['icews14', 'icews18', 'icews0515']

    gcn_layers_list = [1]
    dataset_list = ['icews14']

    gpu = 6

    for gcn_layer in gcn_layers_list:
        for dataset in dataset_list:
            temp_cmd = f'python run_model.py --gpu {gpu} --gcn_layer {gcn_layer} --dataset {dataset}'
            os.system(temp_cmd)

# run_gcn()

