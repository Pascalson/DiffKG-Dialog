import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import timeit
from transformers import T5Tokenizer
from models import *
from utils import make_logdir, get_dataset, get_shared_knowledge_graph
from utils import get_KGs_by_tensorID, get_entities_mappings, truncate_batch
from utils import get_KG_sparse_metrices
import datetime
from nltk.translate.bleu_score import sentence_bleu

from itertools import product
import re
import os
import tqdm
import pdb
import logging
logger = logging.getLogger(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--device', type=str, required=False, default='cuda')
    return parser.parse_args()
    
def get_sentence_entities(sentence:str, entities:list):
    include_ents = []
    for i, ent in enumerate(entities):
        if ent.lower() in sentence.lower():
            include_ents.append(ent.lower())
    return include_ents
    
def span_f1_measure_eval(A, B):
    A_set = set(A)
    B_set = set(B)
    tp = A_set.intersection(B_set)
    fp = B_set - tp
    fn = A_set - tp
    precision = len(tp)/(len(tp)+len(fp))
    recall    = len(tp)/(len(tp)+len(fn))
    if precision==0: 
        return 0
    return 2*recall*precision/(recall+precision)

### Tokenizer ###

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
def basic_tokenize(sentence):
    sentence = _WORD_SPLIT.sub(" ",sentence)
    return sentence.strip().split()

def initialize_model(args):
    print("Loading tokenizer and model from {}".format(args.model_dir))
    if args.device not in ['cuda', 'cpu']:
        raise ValueError('device type should be one of cuda or cpu')
    device = torch.device(args.device)

    model = DiffkgT5.from_pretrained(args.model_dir).to(device)
    tokenizer = model.tokenizer
    return model, tokenizer, device


def main_batch(args):
    model, tokenizer, device = initialize_model(args)
    model.eval()

    KG = get_shared_knowledge_graph(os.path.join(args.data_dir,"KG"))
    args.N_r = len(KG['rels_dict']) 
    N_E = len(KG["ents_dict"])
    padding = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    model.cache_special_token_ids([padding, eos])
    output_id = 2# the logits' position in model's outputs

    print('Reading input files from {}'.format(args.data_dir))
    test_set = get_dataset(tokenizer, args.data_dir, "test", KG)
    test_loader = model.get_dataloader(test_set, args.batch_size, N_E, shuffle=False)
    KG_sparse_Ms = get_KG_sparse_metrices(KG, args.device)
    ents_map = get_entities_mappings(list(KG["ents_dict"].keys()), tokenizer, model._model.config.vocab_size).to(args.device)
    model.build_entities_embeddings(ents_map)

    ents_list = [k for k,v in sorted(KG['ents_dict'].items(), key=lambda x: x[1])]
    rels_list = [k for k,v in sorted(KG['rels_dict'].items(), key=lambda x: x[1])]

    fout = open(os.path.join(args.model_dir,"visualization.txt"), "w")
    batch_index = 1
    labels = []
    rels_seqs = []
    rels_seqs_labels = []
    ents_seqs = []
    ents_seqs_labels = []

    def run_model(batch, KGs, **generator_args):
        with torch.no_grad():
            res, log_probs, init_ents, rels, out_ents = model(batch, KGs, generator_args=generator_args)
        res = [y[:y.index(eos)] if eos in y else y for y in res.tolist()]
        preds = tokenizer.batch_decode(res, skip_special_tokens=True)
        rels_seqs_labels.extend(batch[5].tolist())
        ents_seqs_labels.extend(batch[6].tolist())
        for j in range(len(preds)):
            label = tokenizer.decode(batch[output_id][j], skip_special_tokens=True)
            labels.append(label)
            rels_seq, ents_seq = [],[]
            for h in range(model.max_hops_num):
                values, indices = torch.topk(rels[j,h],5)
                rels_seq.append(list(zip(indices.tolist(),values.tolist())))
                values, indices = torch.topk(out_ents[j,h],25)
                ents_seq.append(indices.tolist())
            # count the relation sequence probability
            seqs = product(*rels_seq)
            rels_seq = {}
            for relation_seq in seqs:
                keys, values = zip(*relation_seq)
                rels_seq[keys] = np.prod(values)
            sorted_rels_seq =  [k for k, v in sorted(rels_seq.items(), key=lambda item: item[1], reverse=True)]
            rels_seqs.append(sorted_rels_seq)
            ents_seqs.append(ents_seq)
            if args.visualize:
                fout.write("\nID{}:\n".format((batch_index-1)*args.batch_size+j))
                fout.write("INPUT\n")
                fout.write("{}\n".format(tokenizer.decode(batch[0][j], skip_special_tokens=True)))
                fout.write("OUTPUT\n")
                fout.write("{}\n".format(label))
                fout.write("INIT ENTS\n")
                values, indices = torch.topk(init_ents[j],5)
                fout.write("{}\n".format([(ents_list[i],v) for i, v in zip(indices.tolist(), values.tolist())]))
                fout.write("RELATIONS & OUT ENTS\n")
                for h in range(model.max_hops_num):
                    values, indices = torch.topk(rels[j,h],5)
                    fout.write("Relations: {}\n".format([(rels_list[i],'{:.4f}'.format(v)) for i, v in zip(indices.tolist(), values.tolist())]))
                    values, indices = torch.topk(out_ents[j,h],5)
                    fout.write("Combinations: {}\n".format([(ents_list[i],'{:.4f}'.format(v)) for i, v in zip(indices.tolist(), values.tolist())]))
                    fout.write("===\n")
                fout.write("PREDICTION\n")
                fout.write("{}\n".format(preds[j]))
        return preds, log_probs

    timer_accumulator = 0
    evaluation_results = {}
    predictions = []
    for batch in tqdm.tqdm(test_loader):
        tic=timeit.default_timer()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        prediction, log_probs = run_model(batch, KG_sparse_Ms, max_length=20, do_sample=False)
        predictions.extend(prediction)
        toc=timeit.default_timer()
        sample_inference_time = toc - tic
        timer_accumulator +=  sample_inference_time
    fout.close()
    compute_metrics(predictions, labels, ents_list, rels_seqs, rels_seqs_labels, ents_seqs, ents_seqs_labels)
    print("Average latency on {} was: {}".format(
        args.device, round(timer_accumulator/len(predictions),2)))

def compute_metrics(predictions, labels, ents_list, rels, rels_labels, ents, ents_labels):
    total_samples = len(predictions)
    bleu1_score, bleu2_score, bleu3_score, bleu4_score = 0,0,0,0
    entities_F1 = 0
    rpath_Recalls = {1:0,3:0,5:0,10:0,25:0}
    
    # relations recalls
    for rel_seq, rel_seq_label in zip(rels, rels_labels):
        for recall_num in rpath_Recalls.keys():
            if tuple(rel_seq_label) in rel_seq[:recall_num]:
                rpath_Recalls[recall_num] += 1

    for pred, label in zip(predictions, labels):
        # compute BLEU scores
        candidate = basic_tokenize(pred)
        references = [basic_tokenize(label)]
        bleu1_score += sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
        bleu2_score += sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0))
        bleu3_score += sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0))
        bleu4_score += sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        # compute entities F1
        target_ents = get_sentence_entities(label, ents_list)
        pred_ents = get_sentence_entities(pred, ents_list)
        entities_F1 += span_f1_measure_eval(target_ents, pred_ents)

    evaluation_results = {}
    evaluation_results["BLEU-1"] = bleu1_score / len(predictions)
    evaluation_results["BLEU-2"] = bleu2_score / len(predictions)
    evaluation_results["BLEU-3"] = bleu3_score / len(predictions)
    evaluation_results["BLEU-4"] = bleu4_score / len(predictions)
    evaluation_results["Entities-F1"] = entities_F1 / len(predictions)
    if len(rels) > 0:
        for k,v in rpath_Recalls.items():
            evaluation_results["Overall RelPath_R{}".format(k)] = v/len(predictions)
    print(evaluation_results)

    with open(os.path.join(args.model_dir,"evaluation.json"), "w") as fp:
        json.dump(evaluation_results, fp, indent=4)

if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cuda':
        main_batch(args)
    elif args.device == 'cpu':
        args.batch_size = 1
        main_batch(args)
    else:
        raise ValueError('device type should be cuda')
