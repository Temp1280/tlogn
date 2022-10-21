# -*- coding: utf-8 -*-

from os.path import join

import numpy as np

from utils.dataset_new import tKGDataset, tkg_collate_fn, tkg_collate_fn2


data_set = 'icews0515'
data_dir = join('../data', data_set)
tkg_dataset_train = tKGDataset(data_dir, data_type='train')

id_2rel = {}
id_2ent = {}
id_2ts = {}
entity2id = tkg_dataset_train.entity2id
rel2id = tkg_dataset_train.relation2id
ts2id = tkg_dataset_train.ts2id
len_rel = len(rel2id)
for key in entity2id.keys():
    id_2ent[entity2id[key]] = key
for key in rel2id.keys():
    id_2rel[rel2id[key]] = key
    id_2rel[rel2id[key]+len_rel] = key + '__reverse'
for key in ts2id.keys():
    id_2ts[ts2id[key]] = key

print('XXXXXX')


def get_neighbors_facts(ent, ts1, ts2):
    neighbors = neighbors_dic[ent]
    facts = tkg_dataset_train.all_facts[neighbors]
    ts_s = facts[:,-1]
    a = ts_s>= ts1
    b = ts_s<ts2
    # print(a&b)
    facts = facts[a&b]
    return facts

def print_result(s,r,o,ts, neighbors3):
    print(id_2ent[s], id_2rel[r], id_2ent[o], id_2ts[ts])
    for item in neighbors3:
        # if o == item[10]:
        #     print(id_2ent[item[0]], id_2rel[item[1]], id_2ent[item[2]], id_2ts[item[3]],
        #           id_2ent[item[4]], id_2rel[item[5]], id_2ent[item[6]], id_2ts[item[7]],
        #           id_2ent[item[8]], id_2rel[item[9]], id_2ent[item[10]], id_2ts[item[11]])

        print(id_2ent[item[0]], id_2rel[item[1]], id_2ent[item[2]], id_2ts[item[3]],
              id_2ent[item[4]], id_2rel[item[5]], id_2ent[item[6]], id_2ts[item[7]],
              id_2ent[item[8]], id_2rel[item[9]], id_2ent[item[10]], id_2ts[item[11]])





neighbors_dic = tkg_dataset_train.Dic_E   # 目前存在的问题是逆向有回路
# hop1_num
n = 0
# for i, fact in enumerate(tkg_dataset_train.test_facts):  # Obama 711
#     s,r,o,ts = fact
#     print(i)
#     # if s != 711:
#     #     continue
#     # print(fact)
#     neighbors1 = get_neighbors_facts(s, 0, ts)
#
#     neighbors2 = []
#     for fact1 in neighbors1:
#         s1, r1, o1, ts1 = fact1
#         tmp_neighbors = get_neighbors_facts(o1, ts1, ts)
#         for item in tmp_neighbors:
#             neighbors2.append([s1, r1, o1, ts1, item[0],item[1],item[2],item[3]])
#     neighbors2 = np.array(neighbors2)
#
#     neighbors3 = []
#     for fact2 in neighbors2:
#         s1, r1, o1, ts1, s2, r2, o2, ts2 = fact2
#         tmp_neighbors = get_neighbors_facts(o2, ts2, ts)
#         for item in tmp_neighbors:
#             neighbors3.append([s1, r1, o1, ts1, s2, r2, o2, ts2, item[0], item[1], item[2], item[3]])
#     neighbors3 = np.array(neighbors3)
#
#     print_result(s,r,o,ts, neighbors3)
#
#
#
#
#     # if s == 711:
#     #     n += 1
#
#     # print(i)
#     if i > 10:
#         break

# print(n)

for i, fact in enumerate(tkg_dataset_train.test_facts):  # Obama 711

    s,r,o,ts = fact

    # if s != 711:
    #     continue

    if i <= 8000:
        continue

    s_ = np.array([s])
    ts_ = np.array([ts])

    parent_indexs, neighbors = tkg_dataset_train.load_neighbors_by_array(s_, ts_, 3, 50, 0.1)
    n_facts1 = tkg_dataset_train.all_facts[neighbors]
    t1 = n_facts1[:, -2]
    ts1 = n_facts1[:, -1]
    # print(parent_indexs, neighbors)

    ts_ = np.array([ts]*len(t1))
    parent_indexs, neighbors = tkg_dataset_train.load_neighbors_by_array(t1, ts_, 3, 50, 0.1, mask_2=True, pre_time=ts1)
    n_facts2 = tkg_dataset_train.all_facts[neighbors]
    t2 = n_facts2[:, -2]
    ts2 = n_facts2[:, -1]
    cur_paths = np.concatenate([n_facts1[parent_indexs], n_facts2], axis=-1)

    ts_ = np.array([ts] * len(t2))
    parent_indexs, neighbors = tkg_dataset_train.load_neighbors_by_array(t2, ts_, 3, 50, 0.1, mask_2=True, pre_time=ts2)
    n_facts3 = tkg_dataset_train.all_facts[neighbors]
    t3 = n_facts3[:, -2]
    ts3 = n_facts3[:, -1]

    cur_paths = np.concatenate([cur_paths[parent_indexs], n_facts3], axis=-1)


    # print(n_facts1.shape,n_facts2.shape,n_facts3.shape)
    # print(cur_paths.shape)
    cur_paths_no = cur_paths[cur_paths[:, -2] != o][:100]
    print(cur_paths_no.shape)
    cur_paths = cur_paths[cur_paths[:,-2]==o][-300:]
    print(cur_paths.shape)
    # print(cur_paths)

    print_result(s, r, o, ts, cur_paths)


    print('---'*100)
    print_result(s, r, o, ts, cur_paths_no)
    break




