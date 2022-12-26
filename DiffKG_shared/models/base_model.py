import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pdb


class DiffKGBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _pointer(self, last_hidden_state):
        x = last_hidden_state.reshape(-1, self._model.config.hidden_size)
        D = last_hidden_state[:,-1,:].unsqueeze(1).repeat(1, last_hidden_state.shape[1], 1)
        D = D.reshape(-1, self._model.config.hidden_size)
        pointers = self.W_p( self.W_q(D) + self.W_v(x) ).reshape(last_hidden_state.shape[0], -1)
        pointers = F.softmax(pointers, dim=1)
        return pointers

    def get_input_embeddings(self, hidden_states):
        return hidden_states[0]

    def get_ents_embeddings(self, ents_map):
        ents_embeds = torch.sparse.mm(ents_map, self._embeddings_w)
        return ents_embeds

    def _pointer_vecs(self, pointers, input_embeddings, attentions):
        vec = torch.sum(pointers.unsqueeze(2) * input_embeddings * attentions.unsqueeze(2), dim=1)
        return vec

    def vec2ent(self, vec, ents_embeds):
        ents_embeds_T = torch.t(ents_embeds/torch.linalg.norm(ents_embeds,dim=1,keepdim=True))
        vec = vec / torch.linalg.norm(vec,dim=1,keepdim=True)
        ents = torch.matmul(vec, ents_embeds_T)
        ents = torch.clamp(ents,min=0,max=1)
        return ents

    def predict_rels(self, D, temperature=1.0):
        rels = self.W_r(D).reshape(D.shape[0], self.max_hops_num, -1)
        rels = F.softmax(rels / temperature, dim=2)
        return rels

    def predict_checks(self, D):
        checks = self.W_c(D).reshape(D.shape[0], self.max_hops_num, -1)
        checks = F.softmax(checks, dim=2)
        return checks

    def follows(self, kgs, rels_seq, init_ent):
        ents = []
        M_h, M_r, M_t = kgs
        ents.append(init_ent)
        for i in range(self.max_hops_num):
            r_i = rels_seq[:,i,:]
            e_i = ents[i]
            walked_ents = torch.sparse.mm(M_t.transpose(1,0), torch.sparse.mm(M_r, r_i.T) * torch.sparse.mm(M_h, e_i.T)).T
            ents.append((walked_ents / (walked_ents.sum(dim=1,keepdim=True)+1e-6)).clone())
        ents = torch.cat([ent.unsqueeze(1) for ent in ents], dim=1)
        return ents
        
    def operates_on_checks(self, D, walk_ents, ents_embeds, temperature=1.0):
        L_cat = F.softmax(self.L(D),dim=1)
        L_op = self.L_d(L_cat).unsqueeze(2)
        ents = torch.zeros(walk_ents.shape).to(self.device)
        for i in range(walk_ents.shape[1]):
            ents_embed = walk_ents[:,i,:].squeeze(1).unsqueeze(2) * ents_embeds.unsqueeze(0)# this step consume the most memory for now
            operated_ents = torch.matmul(ents_embed, L_op[:,:ents_embeds.shape[1],:]).squeeze(2)
            operated_Ds = torch.matmul(D.unsqueeze(1), L_op[:,ents_embeds.shape[1]:,:]).squeeze(2)
            ents[:,i,:] = F.softmax(operated_ents + operated_Ds / temperature, dim=-1)
        return ents, L_cat

    def combine_wock(self, _checks_seq, entities, checks):
        ents = _checks_seq[:,:,0].unsqueeze(2) * entities + _checks_seq[:,:,1].unsqueeze(2) * checks
        return ents
