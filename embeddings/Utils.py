import numpy as np
import pandas as pd
import torch
from torchkge.data_structures import KnowledgeGraph


def to_structure_kg(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
    df = df.drop_duplicates().loc[:, ['h', 't', 'r']]
    kg = KnowledgeGraph(pd.DataFrame(df.values, columns=['from', 'to', 'rel']))
    return kg


def train_test_split_kg(kg, test_size=0.2, type_split='share'):
    if type_split != 'share':
        sizes = kg.n_facts
        train, test = kg.split_kg(
            sizes=[round(sizes * (1 - test_size)), round(sizes * test_size)])
    else:
        train, test = kg.split_kg(share=test_size)
    return train, test


def prepare_full_graph_compare_joie(full_kg, test_kg):
    test_df = test_kg.get_df()
    return KnowledgeGraph(test_df, ent2ix=full_kg.ent2ix, rel2ix=full_kg.rel2ix)


def reindex_kg_idx_enteties_vector(idx_vector, kg, reindex_dict):
    '''функция, которая реиндексирует по словарю example:idx вектор типа head_idx,tail_idx'''
    idx2ent = dict(map(reversed, kg.ent2ix.items()))
    reindex_map_func = np.vectorize(lambda x: reindex_dict.get(idx2ent[x], 0))

    return torch.from_numpy(reindex_map_func(idx_vector.detach().numpy())).long()


def reindex_dict_of_enteties_kg(dict_of, kg, example2idx):
    idx2ent = dict(map(reversed, kg.ent2ix.items()))

    reindex_dict_of = {}

    for key, values in dict_of.items():
        first = example2idx.get(idx2ent[key[0]], 0)
        second = key[1]

        reindex_dict_of[(first, second)] = set(
            map(lambda x: example2idx.get(idx2ent[x], 0), values))

    return reindex_dict_of

def get_relevant_triples(beta, G):
    return set(filter(lambda t: t[1] == beta[1], G))

