import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
    "y_input_ids", "y_attention_mask",
    "labels","path_rels","path_ents",
]

class DiffkgT5(DiffKGBase):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-small")
        self._model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-small")
        self._embeddings_w = self._model.shared.weight
        # relation module
        self.max_hops_num = args.max_hops_num
        self.N_r = args.N_r
        self.W_r = nn.Linear(self._model.config.hidden_size, self.N_r * self.max_hops_num, bias=False)
        self.W_c = nn.Linear(self._model.config.hidden_size, 2 * self.max_hops_num, bias=False)
        # operation matrices
        self.ops_num = 5
        self.L = nn.Linear(self._model.config.hidden_size, self.ops_num, bias=False)
        self.L_d = nn.Linear(self.ops_num, self._model.config.hidden_size * 2, bias=False)

    def resize_token_embeddings(self, new_num_tokens=0):
        self._model.resize_token_embeddings(new_num_tokens=new_num_tokens)

    def cache_special_token_ids(self, special_token_ids):
        self.special_token_ids = special_token_ids
        if isinstance(special_token_ids, list):
            self.padding, self.eos = special_token_ids
        else:
            self.padding = special_token_ids

    def _read_batch(self, batch):
        inputs = {}
        for name, name_batch in zip(MODEL_INPUTS+['slots'], batch):
            inputs[name] = name_batch
        return inputs

    @classmethod
    def from_pretrained(cls, model_dir):
        args_file = os.path.join(model_dir, 'model_training_args.bin')
        args = torch.load(args_file)
        model = cls(args)
        model_state = os.path.join(model_dir, 'pytorch_model.bin')
        state = torch.load(model_state)
        model.load_state_dict(state)
        return model

    def build_entities_embeddings(self, ents_map):
        self._ents_embeddings = self.get_ents_embeddings(ents_map)

    def forward(self, batch, KGs, labels=None, generator_args=None):
        inputs = self._read_batch(batch)

        x_outputs = self._model.encoder( \
            inputs['x_input_ids'],
            attention_mask=inputs['x_attention_mask'],
            output_hidden_states=True,
        )

        _init_ents = inputs['slots']

        # predict relations (and checks; not used now)
        _relations_seq = self.predict_rels(x_outputs.last_hidden_state[:,-1,:], temperature=1.0)
        entities = self.follows(KGs, _relations_seq, _init_ents)
        out_ents = entities[:,1:,:]

        decoder_inputs = self.form_decoder_inputs(inputs, out_ents, self._ents_embeddings)

        if labels is not None:
            outputs = self._model(**decoder_inputs)
            path_loss_fct = nn.NLLLoss()
            rel_loss = path_loss_fct(torch.log(_relations_seq.transpose(1,2)), inputs['path_rels'])
            ent_loss = path_loss_fct(torch.log(out_ents.transpose(1,2)+1e-7), inputs['path_ents'])
            outputs = (rel_loss, ent_loss, outputs)
        else:
            outputs, log_probs = self.sample_sequence(decoder_inputs['inputs_embeds'], decoder_inputs['attention_mask'], **generator_args)
            outputs = (outputs, log_probs) + (_init_ents, _relations_seq, out_ents)
        return outputs


    def sample_sequence(self, input_embeds, attentions, 
            max_length = 20, min_length = 3, do_sample = False, temperature = 1.0):

        _embeddings_w = self._model.shared
        current_output = torch.full((input_embeds.shape[0],1),self.padding).to(self.device)
        log_probs = [0 for _ in range(input_embeds.shape[0])]
        stop_record = [False for _ in range(input_embeds.shape[0])]

        for i in range(max_length):
            decoder_input_ids = current_output
            decoder_input_embeds = _embeddings_w(decoder_input_ids)

            outputs= self._model(inputs_embeds=input_embeds, attention_mask=attentions, decoder_inputs_embeds=decoder_input_embeds)
            logits = outputs[0][:,-1,:].squeeze(1)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if not do_sample else torch.multinomial(probs, 1)
            for j in range(input_embeds.shape[0]):
                if prev[j].item() == self.eos:
                    stop_record[j] = True
                if not stop_record[j]:
                    log_probs[j] += torch.log(probs[j,prev[j].item()]).item()

            current_output = torch.cat([current_output, torch.tensor(prev.tolist()).to(self.device).view(-1,1)], dim=1)

        return current_output, log_probs

    def form_decoder_inputs(self, ori_inputs, entities, ents_embeds, labels=False):
        inputs = {}
        _embeddings_w = self._model.shared

        # Concat the last top10 entities and Compute embeddings of entities and dialogue history
        topk_ents = torch.topk(entities[:,-1,:], 10)
        entities_embeds = torch.matmul(topk_ents.values.unsqueeze(1), ents_embeds[topk_ents.indices])
        input_ids_embeds = _embeddings_w(ori_inputs['x_input_ids'])
        
        # Construct the inputs for decoder
        inputs['inputs_embeds'] = torch.cat([entities_embeds, input_ids_embeds], dim=1)
        inputs['attention_mask'] = torch.cat([torch.full((ori_inputs['x_attention_mask'].shape[0],1), 1).to(self.device), ori_inputs['x_attention_mask']], dim=1)
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

    def get_dataloader(self, data, batch_size, N_E, shuffle=True):

        logger.info("Build inputs and labels")
        dataset = defaultdict(list)

        # convert dialogues and labels into inputs format
        for i, dialog in enumerate(data):
            history = dialog["history"]
            query = dialog["user_query"]
            label = dialog["output"]
            slots = dialog["init_entity"]
            path_rels = dialog["path_relations"]
            path_ents = dialog["path_entities"]
            if len(path_ents) == 0:# the data's issue
                continue
            instance = self.build_input_from_segments(history + [query], label)
            for input_name, input_array in instance.items():
                dataset[input_name].append(input_array)
            dataset['slots'].append(get_init_ents(slots, N_E))
            dataset['path_rels'].append(path_rels[:3])#truncate to maximum 3 hops
            dataset['path_ents'].append(path_ents[:3] if len(path_ents) > 3 else path_ents + [path_ents[-1]] * (3-len(path_ents)))

        logger.info("Pad inputs and convert to Tensor")
        tensor_dataset = []
        padding = self.tokenizer.pad_token_id
        dataset = pad_dataset(dataset, MODEL_INPUTS, padding=padding)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor_dataset.append(tensor)
        tensor_dataset.append(torch.cat(dataset['slots'],dim=0))

        def CustomDataCollator(batch):
            new_batch = []
            for i in range(7):
                new_batch.append(torch.cat([b[i].view(1,-1) for b in batch],dim=0))
            new_batch.append(torch.cat([b[7].view(1,-1) for b in batch],dim=0))
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
        return dataloader
