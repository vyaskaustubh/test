import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
    
        self.linear1 = nn.Linear(self.in_dim, self.out_dim)      
        self.linear2 = nn.Linear(self.in_dim, self.out_dim)      
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, ego_embeddings, A_in):
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)
        # Equation (8) & (9)
        sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
        bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
        embeddings = bi_embeddings + sum_embeddings
        embeddings = self.message_dropout(embeddings)           
        return embeddings


class KGAT(nn.Module):

    def __init__(self, args, n_users, n_entities, n_relations, A_in=None, user_pre_embed=None, item_pre_embed=None):
        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.kge = args.kge_type
        self.attn = args.attn_type

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.lamda = 1e-5
        
        """ Initialising the model parameters based on the type of the knowledge graph embedding model"""
        
        if self.kge == 'TransMS':
            self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
            self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
            if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
                other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
                nn.init.xavier_uniform_(other_entity_embed)
                entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
                self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
            else:
                nn.init.xavier_uniform_(self.entity_user_embed.weight)
            nn.init.xavier_uniform_(self.relation_embed.weight)
        
        elif self.kge == 'TransR':
            self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
            self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
            self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
            if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
                other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
                nn.init.xavier_uniform_(other_entity_embed)
                entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
                self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
            else:
                nn.init.xavier_uniform_(self.entity_user_embed.weight)
            nn.init.xavier_uniform_(self.relation_embed.weight)
            nn.init.xavier_uniform_(self.trans_M)
        
        elif self.kge == 'TransD':
            self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
            self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)    
            self.entity_user_transfer = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
            self.relation_transfer = nn.Embedding(self.n_relations, self.relation_dim)
            if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
                other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
                nn.init.xavier_uniform_(other_entity_embed)
                entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
                self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
                self.entity_user_transfer.weight = nn.Parameter(entity_user_embed)
            else:
                nn.init.xavier_uniform_(self.entity_user_embed.weight)
                nn.init.xavier_uniform_(self.entity_user_transfer.weight)
            nn.init.xavier_uniform_(self.relation_embed.weight)
            nn.init.xavier_uniform_(self.relation_transfer.weight)
        
        elif self.kge == 'Complex':
            self.ent_re_embeddings = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
            self.ent_im_embeddings = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
            self.rel_re_embeddings = nn.Embedding(self.n_relations, self.relation_dim)
            self.rel_im_embeddings = nn.Embedding(self.n_relations, self.relation_dim)
            self.criterion = nn.Softplus()
            if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
                other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
                nn.init.xavier_uniform_(other_entity_embed)
                entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
                self.ent_re_embeddings.weight = nn.Parameter(entity_user_embed)
                self.ent_im_embeddings.weight = nn.Parameter(entity_user_embed)
            else:
                nn.init.xavier_uniform(self.ent_re_embeddings.weight.data)
                nn.init.xavier_uniform(self.ent_im_embeddings.weight.data)
            nn.init.xavier_uniform(self.rel_re_embeddings.weight.data)
            nn.init.xavier_uniform(self.rel_im_embeddings.weight.data)


        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k]))

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         
        return all_embed

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        all_embed = self.calc_cf_embeddings()                       
        user_embed = all_embed[user_ids]                            
        item_pos_embed = all_embed[item_pos_ids]                    
        item_neg_embed = all_embed[item_neg_ids]                    

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1) 

        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.lamda * l2_loss
        return loss
    
    def _transfer(self, e, e_transfer, r_transfer):
        e = e + torch.sum(e * e_transfer, -1, True) * r_transfer
        e_norm = F.normalize(e, p = 2, dim = -1)
        return e_norm

    def calc_kg_loss(self, h, r, pos_t, neg_t):
                                           
        if self.kge == 'TransMS':
            r_embed = self.relation_embed(r)                                                
            h_embed = self.entity_user_embed(h)                                             
            pos_t_embed = self.entity_user_embed(pos_t)                                     
            neg_t_embed = self.entity_user_embed(neg_t)  
            r_t_p = torch.tanh(torch.mul(pos_t_embed, r_embed))
            r_t_n = torch.tanh(torch.mul(neg_t_embed, r_embed))
            r_h = torch.tanh(torch.mul(h_embed, r_embed))
            
            pos = (-1 * torch.mul(r_t_p, h_embed)) + r_embed + torch.mul(h_embed, pos_t_embed) - torch.mul(r_h, pos_t_embed)
            neg = (-1 * torch.mul(r_t_n, h_embed)) + r_embed + torch.mul(h_embed, neg_t_embed) - torch.mul(r_h, neg_t_embed)
            
            pos_score = torch.sum(torch.pow(pos,2), dim=1)
            neg_score = torch.sum(torch.pow(neg,2), dim=1)
        
        if self.kge == 'TransR':
            r_embed = self.relation_embed(r)                                                
            W_r = self.trans_M[r]                                                          

            h_embed = self.entity_user_embed(h)                                            
            pos_t_embed = self.entity_user_embed(pos_t)                                 
            neg_t_embed = self.entity_user_embed(neg_t)                                    

            r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                      
            r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
            r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)          

            # Equation (1)
            pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
            neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)
        
        if self.kge == 'TransD':
            r_embed = self.relation_embed(r)                                                
            h_embed = self.entity_user_embed(h)                                             
            pos_t_embed = self.entity_user_embed(pos_t)                                     
            neg_t_embed = self.entity_user_embed(neg_t)  
            h_transfer = self.entity_user_transfer(h)
            pos_t_transfer = self.entity_user_transfer(pos_t)
            neg_t_transfer = self.entity_user_transfer(neg_t)
            r_transfer = self.relation_transfer(r)
            
            r_mul_h = self._transfer(h_embed, h_transfer, r_transfer)
            r_mul_pos_t = self._transfer(pos_t_embed, pos_t_transfer, r_transfer)
            r_mul_neg_t = self._transfer(neg_t_embed, neg_t_transfer, r_transfer)
            # Equation (1)
            pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)    
            neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)
        
        if self.kge == 'Complex':
            h_re = self.ent_re_embeddings(h)
            h_im = self.ent_im_embeddings(h)
            p_t_re = self.ent_re_embeddings(pos_t)
            p_t_im = self.ent_im_embeddings(pos_t)
            n_t_re = self.ent_re_embeddings(neg_t)
            n_t_im = self.ent_im_embeddings(neg_t)
            r_re = self.rel_re_embeddings(r)
            r_im = self.rel_im_embeddings(r)

            pos_score = -torch.sum(h_re * p_t_re * r_re + h_im * p_t_im * r_re + h_re * p_t_im * r_im - h_im * p_t_re * r_im,-1,)
            neg_score = -torch.sum(h_re * n_t_re * r_re + h_im * n_t_im * r_re + h_re * n_t_im * r_im - h_im * n_t_re * r_im,-1,)
        
        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.lamda * l2_loss
        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        
        if self.kge == 'TransD' or 'TransMS' or 'TransR':
            r_embed = self.relation_embed.weight[r_idx]
            h_embed = self.entity_user_embed.weight[h_list]
            t_embed = self.entity_user_embed.weight[t_list]

            if self.kge == 'TransMS':
                r_h = torch.tanh(torch.mul(h_embed, r_embed))
                r_t = torch.tanh(torch.mul(t_embed, r_embed))
                
                r_mul_h = (-1 * torch.mul(r_t, h_embed)) 
                r = r_embed + torch.mul(h_embed, t_embed) 
                r_mul_t = (-1 * torch.mul(r_h, t_embed))
                
            if self.kge == 'TransD':
                h_transfer = self.entity_user_transfer.weight[h_list]
                t_transfer = self.entity_user_transfer.weight[t_list]
                r_transfer = self.relation_transfer.weight[r_idx]
                
                r_mul_h = self._transfer(h_embed, h_transfer, r_transfer)
                r_mul_t = self._transfer(t_embed, t_transfer, r_transfer)
            
            if self.kge == 'TransR':
                W_r = self.trans_M[r_idx]
                r_mul_h = torch.matmul(h_embed, W_r)
                r_mul_t = torch.matmul(t_embed, W_r)

            v_list = torch.sum(r_mul_t * 1.7159 * torch.tanh((2/3) * (r_mul_h + r_embed)), dim=1)


        if self.kge == 'Complex':
            h_re = self.ent_re_embeddings.weight[h_list]
            h_im = self.ent_im_embeddings.weight[h_list]
            t_re = self.ent_re_embeddings.weight[t_list]
            t_im = self.ent_im_embeddings.weight[t_list]
            r_re = self.rel_re_embeddings.weight[r_idx]
            r_im = self.rel_im_embeddings.weight[r_idx]

            # Equation (4)
            h_embed = torch.add(h_re, h_im)
            t_embed = torch.add(t_re, t_im)
            r_embed = torch.add(r_re, r_im)
            v_list = torch.sum(t_embed * 1.7159 * torch.tanh((2/3) * (h_embed + (r_embed))), dim=1)
        
        if self.attn == 'hybrid':
            x = (torch.norm(h_embed) * torch.norm(t_embed))
            v_2 = torch.div(torch.sum(torch.mul(h_embed, t_embed)), x)
            v_list = torch.mul(v_list,v_2)

        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def calc_score(self, user_ids, item_ids):
        all_embed = self.calc_cf_embeddings()           
        user_embed = all_embed[user_ids]               
        item_embed = all_embed[item_ids]          

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)
