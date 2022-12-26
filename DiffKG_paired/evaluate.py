import argparse
import json
import pandas as pd
import numpy as np
from itertools import chain
import torch
import torch.nn.functional as F
import timeit
from models import *
from utils import make_logdir, get_dataset
import datetime
from nltk.translate.bleu_score import sentence_bleu
from other_measures import moses_multi_bleu

import re
import os
import tqdm
import logging
import pdb

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--only_reasoning_data", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    return parser.parse_args()

### Tokenizer for Evaluation ###

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
def basic_tokenize(sentence):
    sentence = _WORD_SPLIT.sub(" ",sentence)
    return sentence.strip().split()


### F1 scores variants ###
    
def get_sentence_entities(sentence:str, entities:list, extra_ents:list):
    """
    extract entities from sentences given an entity list
    This function might need intensive rules since prior work (the reference data) changed the entities format.
    """
    include_ents = []
    sentence = sentence.lower()
    sentence = sentence.replace(","," ")
    sentence = sentence.replace("?"," ")
    
    for word in sentence.split():
        if word in entities or word in extra_ents:
            include_ents.append(word)

    return list(set(include_ents))


def f1_measure(TP, FP, FN):
    """
    compute F1 given the values of true-positive (TP), false-positive (FP), and false-negative (FN)
    """
    precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
    recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
    F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    return F1

def macro_f1(gold, pred):
    """
    The F1 implementation in GraphDialog
    """
    TP, FP, FN = 0, 0, 0
    for g in gold:
        if g in pred:
            TP += 1
        else:
            FN += 1
    for p in pred:
        if p not in gold:
            FP += 1
    F1 = f1_measure(TP, FP, FN)
    return F1, TP, FP, FN

def span_f1_measure_eval(A, B):
    r"""
    Compute overall F1 scores for semantic forms
    """
    # fixme: using set() is not correct
    template_words = ["with","attribute","value","and","of"]
    A_set = set(a for a in A if a not in template_words)
    B_set = set(b for b in B if b not in template_words)
    tp = A_set.intersection(B_set)
    fp = B_set - tp
    fn = A_set - tp
    precision = len(tp)/(len(tp)+len(fp))
    recall    = len(tp)/(len(tp)+len(fn))
    if precision==0: 
        return 0
    return 2*recall*precision/(recall+precision)

def get_evaluation_buckets(evaluation_results):
    r"""
    Save the SMD-reasoning evaluation results in different question types
    """
    buckets = {'question_type':{}, 'question_attribute':{}}
    for key in evaluation_results:
        if evaluation_results[key]['is_correct'] == 0:
            if evaluation_results[key]['question_type'] in buckets['question_type']:
                if 'false' in buckets['question_type'][evaluation_results[key]['question_type']]:
                    buckets['question_type'][evaluation_results[key]['question_type']]['false'] += 1
                else:
                    buckets['question_type'][evaluation_results[key]['question_type']]['false'] = 1
            else:
                buckets['question_type'][ evaluation_results[key]['question_type']] = {'false':1}

        else:
            if evaluation_results[key]['question_type'] in buckets['question_type']:
                if 'true' in buckets['question_type'][ evaluation_results[key]['question_type']]:
                    buckets['question_type'][ evaluation_results[key]['question_type']]['true'] += 1
                else:
                    buckets['question_type'][ evaluation_results[key]['question_type']]['true'] = 1
            else:
                buckets['question_type'][ evaluation_results[key]['question_type']] = {'true':1}
        if 'avg_span_f1_measure' not in buckets['question_type'][ evaluation_results[key]['question_type']]:
            buckets['question_type'][ evaluation_results[key]['question_type']]['avg_span_f1_measure'] = evaluation_results[key]['span_f1_measure']
        else:
            buckets['question_type'][ evaluation_results[key]['question_type']]['avg_span_f1_measure'] += evaluation_results[key]['span_f1_measure']

    for key in buckets:
        for id_ in buckets[key]:
            if 'true' in buckets[key][id_] and 'false' in buckets[key][id_]:
                buckets[key][id_]['total'] = buckets[key][id_]['true'] + buckets[key][id_]['false']
                buckets[key][id_]['false_percanetage'] = round(100*buckets[key][id_]['false'] / buckets[key][id_]['total'],1)
            elif 'true' in buckets[key][id_]:
                buckets[key][id_]['total'] = buckets[key][id_]['true']
                buckets[key][id_]['false_percanetage'] = 0
            elif 'false' in buckets[key][id_]:
                buckets[key][id_]['total'] =  buckets[key][id_]['false']
                buckets[key][id_]['false_percanetage'] = 1
            buckets[key][id_]['avg_span_f1_measure'] /= buckets[key][id_]['total']

    return buckets


def infer_diffkg_model(model, batch, KGs, ents_map, ents_map_attns, ents_list, rels_list, fout, **generator_args):
    r"""
    Visualization method for DiffKG model
    """
    output_id = 2#TODO: this is according to the current output order of t5 forward(). There is for sure better way to implement this, e.g., using a dict.
    with torch.no_grad():
        res, log_probs, init_ents, rels, checks, out_ents, walk_ents, check_ents = \
            model(batch, KGs, ents_map, ents_map_attns, generator_args=generator_args)
    eos = model.tokenizer.eos_token_id
    res = [y[:y.index(eos)] if eos in y else y for y in res.tolist()]
    preds = model.tokenizer.batch_decode(res, skip_special_tokens=True)
    labels = model.tokenizer.batch_decode(batch[output_id]*(batch[output_id]!=-100), skip_special_tokens=True)
    if fout is not None:
        for j in range(len(preds)):
            fout.write("\nINPUT\n")
            fout.write("{}\n".format(model.tokenizer.decode(batch[0][j], skip_special_tokens=True)))
            used_label = model.tokenizer.decode(batch[output_id][j]*(batch[output_id][j]!=-100), skip_special_tokens=True)
            fout.write("OUTPUT\n")
            fout.write("{}\n".format(used_label))
            fout.write("PREDICTION\n")
            fout.write("{}\n".format(preds[j]))
            fout.write("INIT ENTS\n")
            values, indices = torch.topk(init_ents[j],5)
            fout.write("{}\n".format([(ents_list[i],v) for i, v in zip(indices.tolist(), values.tolist())]))
            fout.write("RELATIONS & OUT ENTS\n")
            for h in range(model.max_hops_num):
                values, indices = torch.topk(rels[j,h],5)
                fout.write("Relations: {}\n".format([(rels_list[i],'{:.4f}'.format(v)) for i, v in zip(indices.tolist(), values.tolist())]))
                values, indices = torch.topk(walk_ents[j,h],5)
                fout.write("Walked Entities: {}\n".format([(ents_list[i],'{:.4f}'.format(v)) for i, v in zip(indices.tolist(), values.tolist())]))
                values, indices = torch.topk(check_ents[j,h],5)
                fout.write("Operated Entities: {}\n".format([(ents_list[i],'{:.4f}'.format(v)) for i, v in zip(indices.tolist(), values.tolist())]))
                values, indices = torch.topk(out_ents[j,h],5)
                fout.write("Combinations: {}\n".format([(ents_list[i],'{:.4f}'.format(v)) for i, v in zip(indices.tolist(), values.tolist())]))
                fout.write("===\n")
            fout.write("CHECKS\n")
            fout.write("{}\n".format(checks[j]))
    return preds, labels, log_probs


def main(args):

    logger.info("Load Trained Model")
    model = DiffkgT5.from_pretrained(args.model_dir).to(args.device)
    model.eval()# set to evaluation mode that turns off randomness, e.g., dropout

    logger.info("Load Test Set")
    test_set = get_dataset(model.tokenizer, args.data_dir, "test", only_reasoning=args.only_reasoning_data)
    test_loader, test_KGs, test_max_N_R = model.get_dataloader(test_set, args.batch_size, shuffle=False)

    # map each entity to its tokens with masks
    _, _, test_kg_path = get_data_paths(args.data_dir, "test")
    ents_map, ents_map_attns = get_flatten_entities_mappings(test_kg_path, model.tokenizer, model._model.config.vocab_size)
    ents_map = ents_map.to(args.device)
    ents_map_attns = ents_map_attns.to(args.device)
    
    # for visualization, read in the entities and relations used in test paired KGs 
    with open(os.path.join(test_kg_path,"entities.json"),"r") as fkg:
        ents_list = json.load(fkg)
    with open(os.path.join(test_kg_path,"relations.json"),"r") as fkg:
        rels_list = json.load(fkg)
        rels_list = ["ToSelf"] + rels_list

    logger.info("Predicting for all test set...")
    vis_fout = open(os.path.join(args.model_dir,"visualization.txt"), "w") if args.visualize else None# file path to save visualization
    timer_accumulator = 0# compute inference time
    evaluation_results = {}
    used_labels = []
    predictions = []

    for batch in tqdm.tqdm(test_loader):
        tic=timeit.default_timer()

        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        KGs = get_KGs_by_tensorID(test_KGs, batch[-1], args.device)# get the paired KG
        prediction, used_label, log_probs = \
            infer_diffkg_model(model, batch, KGs, ents_map, ents_map_attns, ents_list, rels_list, vis_fout, max_length=50, do_sample=False)# forward to get outputs
        predictions.extend(prediction)
        used_labels.extend(used_label)

        toc=timeit.default_timer()
        sample_inference_time = toc - tic
        timer_accumulator +=  sample_inference_time

    if vis_fout is not None:
        vis_fout.close()

    logger.info("Computing Evaluation Metrics...")
    labels = [model.tokenizer.decode(x['output']) for x in test_set][:len(predictions)]
    reasoning_types = [x['reasoning_type'] for x in test_set][:len(predictions)]
    tasks = [x['task'] for x in test_set][:len(predictions)]

    if "reasoning" in args.data_dir:
        evaluation_results = eval_semantic_form(predictions, labels, reasoning_types, tasks, only_reasoning_data=args.only_reasoning_data)
    else:
        ents_list = ['_'.join(e.lower().split()) for e in ents_list]
        with open(os.path.join(args.data_dir, "test.json"),"r") as fin:
            d = json.load(fin)
            ents_labels = [x['entity_label'] for x in d][:len(predictions)]
        evaluation_results = \
            eval_natural_language_form(predictions, labels, reasoning_types, tasks, ents_list, ents_labels)

    with open(os.path.join(args.model_dir, "evaluation.json"), "w") as fout:
        json.dump(evaluation_results, fout, indent=4)
    print("Average latency on {} was: {}".format(
        args.device, round(timer_accumulator/len(predictions),2)))


def eval_natural_language_form(predictions, labels, reasoning_types, tasks, ents_list, ents_labels):
    # initialization for micro F1
    entities_F1, ents_turn_count = 0,0
    schedule_entities_F1, schedule_ents_turn_count = 0,0
    navigate_entities_F1, navigate_ents_turn_count = 0,0
    weather_entities_F1, weather_ents_turn_count = 0,0
    # initialization for macro F1
    TP_all, FP_all, FN_all = 0,0,0
    TP_schedule, FP_schedule, FN_schedule = 0,0,0
    TP_weather, FP_weather, FN_weather = 0,0,0
    TP_navigate, FP_navigate, FN_navigate = 0,0,0

    for pred, label, Rtype, task, gt_ents in zip(predictions, labels, reasoning_types, tasks, ents_labels):
        if "extraction" in Rtype:
            continue
        
        # compute nltk-BLEU and MultiBLEU scores
        candidate = basic_tokenize(pred)
        references = [basic_tokenize(label)]
        moses_multibleu = moses_multi_bleu(np.array(predictions), np.array(labels), lowercase=True)

        # compute entities F1
        target_ents = get_sentence_entities(label, ents_list, gt_ents)
        pred_ents = get_sentence_entities(pred, ents_list, gt_ents)
        if set(target_ents) != set(gt_ents):
            # check if the data loading has issue
            print(label)
            print(pred)
            print(target_ents)
            print(gt_ents)

        if len(target_ents) > 0:
            ent_F1, TP, FP, FN = macro_f1(target_ents, pred_ents)
            TP_all += TP
            FP_all += FP
            FN_all += FN
            entities_F1 += ent_F1
            ents_turn_count += 1
            if "schedule" in task:
                schedule_entities_F1 += ent_F1
                schedule_ents_turn_count += 1
                TP_schedule += TP
                FP_schedule += FP
                FN_schedule += FN
            elif "navigate" in task:
                navigate_entities_F1 += ent_F1
                navigate_ents_turn_count += 1
                TP_navigate += TP
                FP_navigate += FP
                FN_navigate += FN
            elif "weather" in task:
                weather_entities_F1 += ent_F1
                weather_ents_turn_count += 1
                TP_weather += TP
                FP_weather += FP
                FN_weather += FN

    evaluation_results = {}
    evaluation_results["MultiBLEU"] = moses_multibleu
    evaluation_results["Entities-macroF1"] = entities_F1 / ents_turn_count
    evaluation_results["Schedule-Entities-macroF1"] = schedule_entities_F1 / schedule_ents_turn_count
    evaluation_results["Navigate-Entities-macroF1"] = navigate_entities_F1 / navigate_ents_turn_count
    evaluation_results["Weather-Entities-macroF1"] = weather_entities_F1 / weather_ents_turn_count
    evaluation_results["Entities-microF1"] = f1_measure(TP_all,FP_all,FN_all)
    evaluation_results["Schedule-Entities-microF1"] = f1_measure(TP_schedule,FP_schedule,FN_schedule)
    evaluation_results["Navigate-Entities-microF1"] = f1_measure(TP_navigate,FP_navigate,FN_navigate)
    evaluation_results["Weather-Entities-microF1"] = f1_measure(TP_weather,FP_weather,FN_weather)

    print(evaluation_results)
    return evaluation_results


def eval_semantic_form(predictions, labels, reasoning_types, tasks, only_reasoning_data=True):
    r"""
    Evaluate the Exact Match and F1 of semantic formed predictions
    """
    # initialization for exact-match
    true_prediction = 0
    # initialization for F1
    span_f1_measure = 0

    evaluation_results = {}

    for pred, label, Rtype, task in zip(predictions, labels, reasoning_types, tasks):
        if only_reasoning_data:
            if 'no-reasoning-required' in Rtype:
                continue

        answer_tokens = basic_tokenize(label.lower())
        prediction_tokens = basic_tokenize(pred.lower())

        # consider any order of products by sorting the tokens before comparing
        answer_tokens.sort()
        prediction_tokens.sort()

        if prediction_tokens == answer_tokens:
            this_EM = 1
            this_F1 = 1
            true_prediction += this_EM
            span_f1_measure += this_F1
        else:
            this_EM = 0
            this_F1 = span_f1_measure_eval(prediction_tokens, answer_tokens)
            span_f1_measure += this_F1
        if Rtype not in evaluation_results:
            evaluation_results[Rtype] = {"EM":0, "F1":0}
        evaluation_results[Rtype]["EM"] += this_EM
        evaluation_results[Rtype]["F1"] += this_F1
    for Rtype, metrics in evaluation_results.items():
        Rtype_num = len([x for x in reasoning_types if x == Rtype])
        evaluation_results[Rtype]["EM"] /= Rtype_num
        evaluation_results[Rtype]["F1"] /= Rtype_num

    total_samples = len(predictions)
    overall_EM = true_prediction / total_samples
    overall_F1 = span_f1_measure / total_samples
    evaluation_results["Total"] = {"EM":overall_EM, "F1":overall_F1}
    
    print(evaluation_results)
    return evaluation_results


if __name__ == '__main__':
    args = set_args()
    main(args)
