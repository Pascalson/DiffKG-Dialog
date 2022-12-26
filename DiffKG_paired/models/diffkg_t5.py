import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import DiffKGBase
from utils import *

from collections import defaultdict
from itertools import chain
import os


SPECIAL_TOKENS = ["</s>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {
    'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
    'additional_special_tokens': ['<speaker1>', '<speaker2>']
}
MODEL_INPUTS = [
    "x_input_ids", "x_attention_mask",
    "y_input_ids", "y_attention_mask", "labels",
]

class DiffkgT5(DiffKGBase):

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-small")
        self._model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-small")
        # Relation modules
        self.max_hops_num = args.max_hops_num
        self.N_r = args.N_r
        self.W_r = nn.Sequential(
          nn.Linear(self._model.config.hidden_size, self._model.config.hidden_size),
          nn.ReLU(),
          nn.Linear(self._model.config.hidden_size, self.N_r * self.max_hops_num),
        )
        self.W_c = nn.Linear(self._model.config.hidden_size, 2 * self.max_hops_num, bias=False)
        self.L = nn.Linear(self._model.config.hidden_size, self._model.config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens=0):#check
        self._model.resize_token_embeddings(new_num_tokens=new_num_tokens)

    def _read_batch(self, batch):
        inputs = {}
        for name, name_batch in zip(MODEL_INPUTS+['slots','ents_label','KG_ID'], batch):
            inputs[name] = name_batch
        return inputs

    @classmethod
    def from_pretrained(cls, model_dir):
        r"""
        The from_pretrained method for DiffkgT5
        """
        # Read arguments and initialize model with the arguments
        args_file = os.path.join(model_dir, 'model_training_args.bin')
        args = torch.load(args_file)
        model = cls(args)
        model._embeddings = model._model.shared

        # Load the model dict_state
        state = torch.load(os.path.join(model_dir, 'pytorch_model.bin'))
        model.load_state_dict(state)

        return model


    def forward(self, batch, KGs, ents_map, ents_map_attns, labels=None, use_entities_labels=False, generator_args=None):
        r"""
        The main forward function of DiffkgT5
        args:
            KGs: the paired knowledge graphs of this batch.
            ents_map: all the entities map to their tokens and paddings.
            ents_map_attns: the attention mask of ents_map.
        """
        # Reformulate the input batch to dict
        inputs = self._read_batch(batch)

        # Encode dialogue history
        x_outputs = self._model.encoder( \
            inputs['x_input_ids'],
            attention_mask=inputs['x_attention_mask'],
        )
        dialogue_embedding = x_outputs.last_hidden_state[:,-1,:]
        
        # Compute entity embeddings
        _init_ents = inputs['slots']
        self._embeddings = self._model.shared
        _ents_embeddings = self.get_ents_embeddings(ents_map, ents_map_attns)

        # Predict relations and conduct NEXT operations
        _relations_seq = self.predict_rels(dialogue_embedding, temperature=1.0)
        entities = self.NEXT(KGs, _relations_seq, _init_ents)

        # Predict checks and conduct check operations
        _checks_seq = self.predict_checks(dialogue_embedding)
        checks = self.operates_on_checks(dialogue_embedding, entities, _ents_embeddings, temperature=1.0)
        out_ents = self.walk_or_check(_checks_seq, entities[:,1:,:], checks[:,:-1,:])

        # Concat the retrieved information and dialogue histroy in text-level
        decoder_inputs = self.form_decoder_inputs(inputs, out_ents, _ents_embeddings, ents_map, ents_map_attns)

        if labels is not None:
            outputs = self._model(**decoder_inputs)
            if use_entities_labels:
                ent_loss = (-inputs['ents_label']*torch.log(out_ents[:,-1,:]+1e-7)).sum()/(inputs['ents_label'].sum()+1e-7)
                outputs = (ent_loss, outputs)
        else:
            outputs, log_probs = self.sample_sequence(decoder_inputs['inputs_embeds'], decoder_inputs['attention_mask'], **generator_args)
            outputs = (outputs, log_probs) + (_init_ents, _relations_seq, _checks_seq, out_ents, entities[:,1:,:], checks[:,:-1,:])

        return outputs


    def sample_sequence(self, input_embeds, attentions,
            max_length = 20, min_length = 3, do_sample = False, temperature = 1.0):

        _embeddings_w = self._model.shared
        current_output = torch.full((input_embeds.shape[0],1),self.tokenizer.pad_token_id).to(self.device)
        log_probs = [0 for _ in range(input_embeds.shape[0])]
        stop_record = [False for _ in range(input_embeds.shape[0])]

        x_outputs = self._model.encoder( \
            inputs_embeds=input_embeds,
            attention_mask=attentions,
        )

        for i in range(max_length):
            decoder_input_ids = current_output
            decoder_input_embeds = _embeddings_w(decoder_input_ids)

            decoder_outputs = self._model.decoder(
                inputs_embeds=decoder_input_embeds,
                encoder_hidden_states=x_outputs[0],
                encoder_attention_mask=attentions,
            )
            lm_outputs = self._model.lm_head(decoder_outputs[0])

            logits = lm_outputs[:,-1,:].squeeze(1)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if not do_sample else torch.multinomial(probs, 1)
            for j in range(input_embeds.shape[0]):
                if prev[j].item() == self.tokenizer.eos_token_id:
                    stop_record[j] = True
                if not stop_record[j]:
                    log_probs[j] += torch.log(probs[j,prev[j].item()]).item()
            current_output = torch.cat([current_output, torch.tensor(prev.tolist()).to(self.device).view(-1,1)], dim=1)
        return current_output, log_probs


    def form_decoder_inputs(self, ori_inputs, entities, ents_embeds, ents_map, ents_map_attns, labels=False):
        inputs = {}
        _embeddings_w = self._model.shared

        # Concat the last top10 entities and Compute embeddings of entities and dialogue history
        topk_ents = torch.topk(entities[:,-1,:], 10)
        entities_token_ids = ents_map[topk_ents.indices].view(entities.shape[0],-1, ents_map.shape[1])
        entities_embeds = _embeddings_w(entities_token_ids) * topk_ents.values.view(topk_ents.values.shape[0],topk_ents.values.shape[1],1,1)
        entities_embeds = entities_embeds.reshape(entities_embeds.shape[0],-1,entities_embeds.shape[3])
        input_ids_embeds = _embeddings_w(ori_inputs['x_input_ids'])

        # Construct the inputs for decoder
        inputs['inputs_embeds'] = torch.cat([entities_embeds, input_ids_embeds], dim=1)
        entities_attention_mask = ents_map_attns[topk_ents.indices].view(entities.shape[0],-1)
        inputs['attention_mask'] = torch.cat([entities_attention_mask, ori_inputs['x_attention_mask']], dim=1)
        inputs['labels'] = ori_inputs['y_input_ids']

        return inputs


    def build_input_from_segments(self, history, reply):
        x_sequences = [s for i, s in enumerate(history)]
        y_sequence = reply + [self.tokenizer.eos_token_id]

        instance = {}
        instance["x_input_ids"] = list(chain(*x_sequences))
        instance["x_attention_mask"] = [1] * len(instance["x_input_ids"])
        instance["y_input_ids"] = y_sequence
        instance["y_attention_mask"] = [1] * len(instance["y_input_ids"])
        instance["labels"] = y_sequence[1:]
        return instance

    def get_dataloader(self, data, batch_size, shuffle=True):

        logger.info("Build inputs and labels")
        dataset = defaultdict(list)
        KGs = defaultdict(list)
        # tokenize dialogues and sparsify KGs
        max_N_T = max(len(dialog["knowledge_graph"]) for dialog in data)
        max_N_E = max(max(triple[0],triple[2])+1 for dialog in data for triple in dialog["knowledge_graph"])
        max_N_R = max(triple[1]+1 for dialog in data for triple in dialog["knowledge_graph"])
        for i, dialog in enumerate(data):
            kg = dialog["knowledge_graph"]
            history = dialog["history"]
            query = dialog["user_query"]
            label = dialog["output"]
            ents_label = dialog["entities_label"]
            slots = dialog["slots"]
            instance = self.build_input_from_segments(history + [query], label)
            for input_name, input_array in instance.items():
                dataset[input_name].append(input_array)
            dataset['slots'].append(get_init_ents(slots, max_N_E))
            dataset['ents_label'].append(get_init_ents(ents_label, max_N_E))
            KGs['M_h'].append(column_to_sparse([triple[0] for triple in kg], max_N_=max_N_E))
            KGs['M_r'].append(column_to_sparse([triple[1] for triple in kg], max_N_=max_N_R))
            KGs['M_t'].append(column_to_sparse([triple[2] for triple in kg], max_N_=max_N_E))

        logger.info("Pad inputs and convert to Tensor")
        tensor_dataset = []
        dataset = pad_dataset(dataset, MODEL_INPUTS, padding=self.tokenizer.pad_token_id)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor_dataset.append(tensor)
        tensor_dataset.append(torch.cat(dataset['slots'],dim=0))
        tensor_dataset.append(torch.cat(dataset['ents_label'],dim=0))
        tensor_dataset.append(torch.tensor(list(range(len(KGs['M_h'])))))

        def CustomDataCollator(batch):
            new_batch = []
            for i in range(7):
                new_batch.append(torch.cat([b[i].view(1,-1) for b in batch],dim=0))
            new_batch.append(torch.cat([b[7].view(-1,) for b in batch],dim=0))
            new_batch[0] = truncate_batch(new_batch[0], self.tokenizer.pad_token_id)
            new_batch[1] = truncate_batch(new_batch[1], 0)
            new_batch[2] = truncate_batch(new_batch[2], -100)#y_input_ids
            new_batch[3] = truncate_batch(new_batch[3], 0)
            new_batch[4] = truncate_batch(new_batch[4], -100, add_length=1)
            return tuple(new_batch)

        logger.info("Build dataloader")
        dataset = TensorDataset(*tensor_dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=CustomDataCollator, pin_memory=True)

        logger.info("set x (Batch, Seq length): {}".format(dataset.tensors[0].shape))
        logger.info("set y (Batch, Seq length): {}".format(dataset.tensors[2].shape))
        logger.info("max N_T={}, N_E={}, N_R={}".format(max_N_T, max_N_E, max_N_R))
        return dataloader, (KGs['M_h'], KGs['M_r'], KGs['M_t']), max_N_R

