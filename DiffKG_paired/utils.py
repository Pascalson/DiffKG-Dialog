from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket
from itertools import chain

import torch
from transformers import cached_path
from difflib import SequenceMatcher
import random
import pdb

logger = logging.getLogger(__file__)

#################
# Data Utilities
#################

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_data_paths(dataset_dir, partition):
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    dataset_path = os.path.join(dataset_dir, partition+'.json')
    kg_path = os.path.join(dataset_dir, partition+'_KGs')
    return dataset_name, dataset_path, kg_path

def get_dataset(tokenizer, dataset_dir, partition, only_reasoning=False, shuffle_kg=False, exclude_extraction=False):

    # Set up the data paths
    dataset_name, dataset_path, kg_path = get_data_paths(dataset_dir, partition)
    dataset_cache = '_'.join(["cache", dataset_name, partition, type(tokenizer).__name__])

    # Reload from cache if exists
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Load json data from %s", dataset_path)
        # Load data
        data_file = cached_path(dataset_path)
        with open(data_file, "r", encoding="utf-8") as f:
            dataset_json = json.loads(f.read())

        # Load entities in the KGs and make each entity a word
        with open(os.path.join(kg_path,"entities.json"),"r") as fkg:
            entities = json.load(fkg)
        entities = [e.lower() for e in entities]
        ents_dict = {e:i for i,e in enumerate(entities)}

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            return list(tokenize(o) for o in obj)

        def read_KG(idx, dialogue):
            r"""
            read a paired KG of the given dialogue's ID
            """
            with open(os.path.join(kg_path,"D{}_ids.json".format(idx)), "r") as fkg:
                indexed_kg = json.load(fkg)

            # Identify which part of the KG should not exist
            if "delete" in dialogue.keys():
                delete_id = dialogue["delete"]
            else:
                delete_id = -100

            # Build KG
            kg = []
            for item in indexed_kg:
                if item[0] != delete_id:
                    kg.append((item[1],item[2]+1,item[3]))
                    if (item[1],0,item[1]) not in kg:
                        kg.append((item[1],0,item[1]))
                    if (item[3],0,item[3]) not in kg:
                        kg.append((item[3],0,item[3]))
            kg_triples = [list(triple) for triple in kg]

            if shuffle_kg:#For analysis
                random.shuffle(kg_triples)

            return kg_triples

        def get_slots_ids(slots):#re-read
            id_list = []
            for slot in slots:
                slot = slot.lower().strip()
                if slot in entities:
                    id_list.append(ents_dict[slot])
                else:
                    for j, e in enumerate(entities):
                        if similar(slot,e) > 0.8:
                            id_list.append(j)
            return id_list

        def get_entities_label(sentence):
            r"""
            Identify which entities exist in the ground-truth response.
            """
            label = []
            for i, ent in enumerate(entities):
                if ent.lower() in sentence.lower():
                    label.append(i)
            return label


        # Build dataset
        logger.info("Tokenize and encode the dataset")
        dataset = []
        for i, dialogue in enumerate(dataset_json):

            kg = read_KG(dialogue["dialogue_id"], dialogue)
            user_query = dialogue['user_query'][1]

            dataset.append({
                'task':dialogue['task'],
                'reasoning_type':dialogue['reasoning_type'] if 'reasoning_type' in dialogue else 'not labeled',
                'history':tokenize([t[1] for t in dialogue['history']]),
                'user_query':tokenize(user_query),
                'output':tokenize(dialogue['output']),
                'knowledge_graph':kg,
                'entities_label':get_entities_label(dialogue['output']),
                'slots':get_slots_ids(dialogue['slots']) if 'slots' in dialogue else get_slots_ids(dialogue['init_entities']),
                # keys 'slots' and 'init_entities' refer to the same thing, only the name is different in different processed data 
            })

        torch.save(dataset, dataset_cache)

    if only_reasoning:
        dataset = [d for d in dataset if d['reasoning_type'] != "no-reasoning-required"]
    if exclude_extraction:
        dataset = [d for d in dataset if d['reasoning_type'] != "extraction"]

    return dataset

def truncate_batch(batch, padding, align='left', add_length=0):#shared, the only difference is add_length
    trunc = [[x for x in s if x != padding] for s in batch.tolist()]
    max_l = max(len(s) for s in trunc) + add_length
    trunc = [s[:(max_l)] if align=='left' else s[-(max_l):] for s in batch.tolist()]
    return torch.tensor(trunc)

def pad_dataset(dataset, MODEL_INPUTS, padding=0):# shared
    for name in MODEL_INPUTS:
        max_l = max([len(x) for x in dataset[name]])
        if "attention_mask" in name:
            pad_type = 0
        elif "labels" in name:
            pad_type = -100
            max_l += 1# the only difference
        elif "y_input_ids" in name:
            pad_type = -100
        else:
            pad_type = padding
        dataset[name] = [x + [pad_type] * (max_l - len(x)) for x in dataset[name]]
    return dataset
    
def column_to_sparse(triple_column, max_N_T=None, max_N_=None):# shared
    N_T = max_N_T if max_N_T is not None else len(triple_column)
    N_ = max_N_ if max_N_ is not None else max(triple_column)+1
    i = [list(range(len(triple_column))), triple_column] 
    v = [1 for _ in range(len(triple_column))]
    sparse_m = torch.sparse.FloatTensor(
        torch.LongTensor(i),
        torch.FloatTensor(v),
        torch.Size([N_T,N_]))
    return sparse_m

def get_entities_mappings(kg_path, tokenizer, vocab_size):# shared
    with open(os.path.join(kg_path,"entities.json"),"r") as fkg:
        entities = json.load(fkg)
    mappings = []
    for e in entities:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e))
        e_mapping = torch.zeros((vocab_size,))# the only difference is the vocab_size
        e_mapping[ids] = 1
        mappings.append(e_mapping.view(1,-1))
    return torch.cat(mappings,dim=0)

def get_flatten_entities_mappings(kg_path, tokenizer, vocab_size):#shared
    with open(os.path.join(kg_path,"entities.json"),"r") as fkg:
        entities = json.load(fkg)
    mappings = []
    for e in entities:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e))
        mappings.append(ids)
    max_l = max(len(ent_map) for ent_map in mappings)
    mappings = [ent_map + [0] * (max_l-len(ent_map)) for ent_map in mappings]
    masks = [[1] * len(ent_map) + [0] * (max_l-len(ent_map)) for ent_map in mappings]
    return torch.tensor(mappings), torch.tensor(masks)

def get_KGs_by_tensorID(KGs, tensorID, device):#shared
    out_KGs = []
    for i in tensorID.tolist():
        out_KGs.append((KGs[0][i].to(device), KGs[1][i].to(device), KGs[2][i].to(device)))
    return out_KGs

def get_init_ents(ids, N_E):#shared
    vec = torch.zeros((N_E,))
    vec[ids] = 1.
    return vec.view(1,-1)


###################
# Other Utilities
###################

def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = model_name + '_' + current_time
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir

