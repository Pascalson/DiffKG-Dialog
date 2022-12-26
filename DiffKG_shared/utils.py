from datetime import datetime
import json
import pandas as pd
import logging
import os
import tarfile
import tempfile
import socket
from itertools import chain

import torch
from transformers import cached_path
from difflib import SequenceMatcher
import pdb

logger = logging.getLogger(__file__)

#################
# Data Utilities
#################

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_data_paths(dataset_dir, partition):
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    dataset_path = os.path.join(dataset_dir, partition+'.csv')
    return dataset_name, dataset_path

def get_dataset(tokenizer, dataset_dir, partition, KG, dataset_cache=None):
    dataset_name, dataset_path = get_data_paths(dataset_dir, partition)
    dataset_cache = '_'.join(["cache", dataset_name, partition, type(tokenizer).__name__])
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Load csv data from %s", dataset_path)
        data_file = cached_path(dataset_path)
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            return list(tokenize(o) for o in obj)

        df = pd.read_csv(dataset_path, converters={'Messages':json.loads})

        logger.info("Tokenize and encode the dataset")
        dataset = []
        for dialogue in df['Messages']:
            history = []
            prev = None
            if 'metadata' in dialogue[0]:
                continue
            for item in dialogue:
                if 'metadata' in item:
                    if 'path' in item['metadata']:
                        try:
                            path = item['metadata']['path'][1]
                        except:
                            pdb.set_trace()
                        rels = []
                        ents = []
                        try:
                            for triple in path:
                                if triple[1].strip().lower() == "~author":
                                    rels.append(KG['rels_dict']["~written_by"])
                                else:
                                    rels.append(KG['rels_dict'][triple[1].strip().lower()])
                                ents.append(KG['ents_dict'][triple[2].strip().lower()])
                            init_ent = [KG['ents_dict'][path[0][0].strip().lower()]]
                            prev = "path"
                        except:
                            pdb.set_trace()
                elif 'message' in item:
                    user_query = item['message']
                    if prev == "message":
                        path = []
                        init_ent = []
                        rels = []
                        ents = []
                        prev = "path"
                    if prev == "path":
                        dataset.append({
                            'history':tokenize(history[:-1]),
                            'user_query':tokenize(history[-1]),
                            'output':tokenize(user_query),
                            'init_entity':init_ent,
                            'path_relations':rels,
                            'path_entities':ents,
                        })
                    history.append(user_query)
                    prev = "message"
                    
        if dataset_cache:
            torch.save(dataset, dataset_cache)

    return dataset

def get_shared_knowledge_graph(kg_path):
    with open(kg_path+"_entities.txt","r") as fin:
        ents_list = fin.readlines()
    ents_dict = {v.strip().lower():i for i, v in enumerate(ents_list)}
    with open(kg_path+"_relations.txt","r") as fin:
        rels_list = fin.readlines()
        rels_list = ["ToSelf"] + rels_list# add ToSelf to not keep walking on the graph
    rels_dict = {v.strip().lower():i for i, v in enumerate(rels_list)}
    with open(kg_path+"_triples.txt","r") as fin:
        triples_list = fin.readlines()
    triples_ids = []
    for triple in triples_list:
        h,r,t = triple.split('\t')
        triples_ids.append((ents_dict[h.strip().lower()], rels_dict[r.strip().lower()], ents_dict[t.strip().lower()]))
    # add ToSelf triples
    for i, ents in enumerate(ents_list):
        triples_ids.append((i,0,i))
    KG = {"triples":triples_ids,"ents_dict":ents_dict, "rels_dict":rels_dict}
    return KG


def truncate_batch(batch, padding, align='left', add_length=0):
    trunc = [[x for x in s if x != padding] for s in batch.tolist()]
    max_l = max(len(s) for s in trunc) + add_length
    trunc = [s[:(max_l)] if align=='left' else s[-(max_l):] for s in batch.tolist()]
    return torch.tensor(trunc)

def pad_dataset(dataset, MODEL_INPUTS, padding=0):
    for name in MODEL_INPUTS:
        max_l = max([len(x) for x in dataset[name]])
        if "attention_mask" in name:
            pad_type = 0
        elif "labels" in name:
            pad_type = -100
            max_l += 1
        elif "path_rel" in name:
            pad_type = 0
            max_l = 3
        else:
            pad_type = padding
        dataset[name] = [x + [pad_type] * (max_l - len(x)) for x in dataset[name]]
    return dataset
    
def column_to_sparse(triple_column, max_N_T=None, max_N_=None):
    N_T = max_N_T if max_N_T is not None else len(triple_column)
    N_ = max_N_ if max_N_ is not None else max(triple_column)+1
    i = [list(range(len(triple_column))), triple_column] 
    v = [1 for _ in range(len(triple_column))]
    sparse_m = torch.sparse.FloatTensor(
        torch.LongTensor(i),
        torch.FloatTensor(v),
        torch.Size([N_T,N_]))
    return sparse_m

def get_entities_mappings(entities, tokenizer, vocab_size):#one major difference from paired version
    sparse_indices = [[],[]]
    sparse_values = []
    for i, e in enumerate(entities):
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e))
        sparse_indices[0].extend([i for _ in ids])
        sparse_indices[1].extend(ids)
        sparse_values.extend([1/len(ids) for _ in range(len(ids))])
    sparse_m = torch.sparse.FloatTensor(
        torch.LongTensor(sparse_indices),
        torch.FloatTensor(sparse_values),
        torch.Size([len(entities),vocab_size]))
    return sparse_m

def get_KGs_by_tensorID(KGs, tensorID, device):
    out_KGs = []
    for i in tensorID.tolist():
        out_KGs.append((KGs[0][i].to(device), KGs[1][i].to(device), KGs[2][i].to(device)))
    return out_KGs

def get_init_ents(ids, N_E):
    if len(ids) == 0:
        vec = torch.ones((N_E,))
    else:
        vec = torch.zeros((N_E,))
        vec[ids] = 1.
    return vec.view(1,-1)

def get_KG_sparse_metrices(KG, device):
    metrices = {}
    metrices['M_h'] = column_to_sparse([triple[0] for triple in KG["triples"]], max_N_ = len(KG["ents_dict"]))
    metrices['M_r'] = column_to_sparse([triple[1] for triple in KG["triples"]], max_N_ = len(KG["rels_dict"]))
    metrices['M_t'] = column_to_sparse([triple[2] for triple in KG["triples"]], max_N_ = len(KG["ents_dict"]))
    return metrices['M_h'].to(device), metrices['M_r'].to(device), metrices['M_t'].to(device)



###################
# Other Utilities
###################

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = model_name + '_' + current_time
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
