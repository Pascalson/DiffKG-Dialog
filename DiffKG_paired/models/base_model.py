import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffKGBase(nn.Module):
    """
    The base class for the proposed DiffKG method.
    Other models inherit this class to reuse the same functions.
    D: dialogue history embedding
    """
    def __init__(self):
        super().__init__()

    def _pointer(self, last_hidden_state):
        """
        This function requires the model having self.W_p, self.W_q, and self.W_v
        to compute pointers towards the tokens in dialogue history.
        self.W_q (encoder_hidden_size x self-defined size)
            => transforming the dialogue embedding D into a "query" vector
        self.W_v (encoder_hidden_size x self-defined size)
            => transforming each token into a "value" vector
        self.W_p (self-defined size x 1)
            => summing "query" and "value" to one scalar
        Return the pointers (batch_size x sequence_length)
        """
        x = last_hidden_state.reshape(-1, self._model.config.hidden_size)
        D = last_hidden_state[:,-1,:].unsqueeze(1).repeat(1, last_hidden_state.shape[1], 1)
        D = D.reshape(-1, self._model.config.hidden_size)
        pointers = self.W_p( self.W_q(D) + self.W_v(x) ).reshape(last_hidden_state.shape[0], -1)
        pointers = F.softmax(pointers, dim=1)
        return pointers

    def _pointer_vecs(self, pointers, input_embeddings, attentions):
        """
        Sum and multiply pointers (batch_size x sequence_length) with 
            the correspondent tokens embeddings (batch_size x sequence_length x encoder_hidden_size)
        Return (batch_size x encoder_hidden_size)
        """
        vec = torch.sum(pointers.unsqueeze(2) * input_embeddings * attentions.unsqueeze(2), dim=1)
        return vec

    def get_ents_embeddings(self, ents_map, ents_map_attns):
        """
        get the entities embeddings when the given ents_map is the tokens ids
        ents_map: (N_E x max_entities_tokens_num)
        ents_map_attns: (N_E x max_entities_tokens_num)
        return (N_E x max_entities_tokens_num x hidden_size)
        """
        ents_embeds = self._embeddings(ents_map) * ents_map_attns.unsqueeze(2)
        return ents_embeds

    def predict_rels(self, D, temperature=1.0):
        rels = self.W_r(D).reshape(D.shape[0], self.max_hops_num, -1)
        rels = F.softmax(rels / temperature, dim=2)
        return rels

    def predict_checks(self, D):
        checks = self.W_c(D).reshape(D.shape[0], self.max_hops_num, -1)
        checks = F.softmax(checks, dim=2)
        return checks

    def NEXT(self, kgs, rels_seq, init_ent):
        """
        Compute the next entities for each hop
        At each hop, the entities probabilities are normalized
        """
        ents = []
        for b in range(rels_seq.shape[0]):
            M_h, M_r, M_t = kgs[b]
            ents.append([init_ent[b].view(1,-1)])
            for i in range(self.max_hops_num):
                r_i = rels_seq[b,i,:].view(-1,1)
                e_i = ents[b][i].view(-1,1)
                walked_ents = torch.sparse.mm(M_t.transpose(1,0),torch.sparse.mm(M_r, r_i)*torch.sparse.mm(M_h, e_i)).view(1,-1)
                ents[b].append((walked_ents / (walked_ents.sum()+1e-6)).clone())
        ents = torch.cat([torch.cat(ents_b, dim=0).unsqueeze(0) for ents_b in ents], dim=0)
        return ents
        
    def operates_on_checks(self, D, walk_ents, ents_embeds, temperature=1.0):
        #ents_embeds: N_E x max_entities_tokens_num x hidden_size
        L_op = self.L(D).view(D.shape[0], 1, 1, 1, D.shape[1])#batch_size x 1 x 1 x 1 x hidden_size
        L_op = L_op.repeat(1,walk_ents.shape[2], ents_embeds.shape[1], 1, 1)#batch_size x N_E x max_entities_tokens_num x 1 x hidden_size
        ents = torch.zeros(walk_ents.shape).to(self.device)
        for i in range(walk_ents.shape[1]):
            ents_embed = walk_ents[:,i,:].view(walk_ents.shape[0], walk_ents.shape[2],1,1) * ents_embeds.unsqueeze(0)#batch_size x N_E x max_entities_tokens_num x hidden_size
            operated = torch.matmul(L_op, ents_embed.unsqueeze(4))#batch_size x N_E x max_entities_tokens_num x 1 x 1
            operated = operated.squeeze()#batch_size x N_E x max_entities_tokens_num
            operated_ents = torch.sum(operated, dim=2)#batch_size x N_E
            ents[:,i,:] = F.softmax(operated_ents / temperature, dim=-1)#batch_size x N_E
        return ents

    def walk_or_check(self, _checks_seq, entities, checks):
        combined = torch.cat([entities.unsqueeze(2), checks.unsqueeze(2)],dim=2)
        ents = torch.matmul(_checks_seq.unsqueeze(2), combined).squeeze(2)
        return ents
