import math

import torch
#from torch._C import INSERT_FOLD_PREPACK_OPS, Node
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
class Position_Game(nn.Module):
   def __init__(self, n_edges, sample_size, state_dim, n_agents):
       super().__init__()
       print(n_edges,sample_size)
       #self.W_line = nn.Parameter(torch.ones(n_edges).cuda())
       self.W_line = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, n_edges),
       )
       self.W = None
       self.sample_size = sample_size
       self.n_ = n_edges
       self.n_agents = n_agents
       self.state_dim = state_dim
   def marginal_utility_function(self, batch_size, node_features, new_hg, states, hypergraph):
       # self.W = torch.diag_embed(self.W_line)  # 超边的权重（e_nodes,e_nodes)
       # self.W = self.W.unsqueeze(0).unsqueeze(0).expand(batch_size, self.sample_size, -1, -1)
       self.W = torch.diag_embed(self.W_line(states.reshape(-1,self.state_dim)))# 超边的权重（n_e,n_e)
       self.W = self.W.unsqueeze(1).expand(-1, self.sample_size, -1, -1)
       self.W = self.W.reshape(-1, self.n_, self.n_)
       B_inv = torch.sum(new_hg.detach(), dim=-2)  # 得到每条边对应结点的个数（b*n_s, 1, n_e）
       B_inv = torch.diag_embed(B_inv)  # 每条边对应结点数的对角矩阵（b*n_s, n_e, n_e）
       softmax_w = torch.abs(self.W).detach()
        # (b*n_s, n_e, n_e)
       D_inv = torch.matmul(new_hg.detach(), softmax_w).sum(dim=-1)  # 加权度矩阵(b*n_s, n, 1)
       D_inv = torch.diag_embed(D_inv)  # 每条边对应节点数加权的对角矩阵(b*n_s, n, n)
       D_inv = D_inv ** (-0.5)
       B_inv = B_inv ** (-1)
       D_inv[D_inv == float('inf')] = 0
       D_inv[D_inv == float('nan')] = 0
       B_inv[B_inv == float('inf')] = 0
       B_inv[B_inv == float('nan')] = 0
       A = torch.bmm(D_inv, new_hg)  # (b*n_s, n, n_e)
       A = torch.matmul(A, torch.abs(self.W))  # (b*n_s, n, n_e)
       A = torch.bmm(A, B_inv)  # (b*n_s, n, n_e)
       X = torch.bmm(A.transpose(-2, -1), \
                     node_features.unsqueeze(1).expand(batch_size, self.sample_size, node_features.shape[-2],
                                                       node_features.shape[-1]) \
                     .reshape(-1, node_features.shape[-2], node_features.shape[-1]))  # (b*n_s, e_nodes, 1)
       return X.reshape(batch_size, self.sample_size, -1, 1)  # (b, n_s, n_e, 1)

   def get_new_hg(self, hyper_graph, grand_coalitions, subcoalition_map):
       batch_size = hyper_graph.size(0)
       grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_, self.n_,
                                                                hyper_graph.size(-2))  # shape = (b, n_s, n_e, n_e, n)
       new_hg = hyper_graph.transpose(-2, -1).unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_,
                                                                               self.n_, hyper_graph.size(-2)).gather(
           3, grand_coalitions)  # shape = (b, n_e, n) -> (b, n_s, 1, n_e, n) -> (b, n_s, n_e, n_e, n)
       map = subcoalition_map.unsqueeze(-1).float()  # shape = (b, n_s, n_e, n_e, 1)
       new_hg = new_hg * map
       new_hg = torch.max(new_hg, dim=-2)[0].reshape(-1, new_hg.shape[-2],
                                                     new_hg.shape[-1])  # shape = (b, n_s, n_e, n) -> (b*n_s, n_e, n)
       return new_hg.transpose(-2, -1)  # shape = (b*n_s, n_e, n)

   def forward(self, node_features, hyper_graph, states, subcoalition_map, grand_coalitions, is_max=None):
       batch_size = hyper_graph.size(0)
       new_hg = self.get_new_hg(hyper_graph, grand_coalitions, subcoalition_map)
       if is_max is not None:
           is_max = torch.matmul(new_hg.transpose(-2, -1), is_max.float())
           is_max = (is_max == new_hg.transpose(-2,-1).sum(dim=-1,keepdim=True)).float()
       X = self.marginal_utility_function(batch_size, node_features, new_hg, states, hyper_graph)
       if is_max is not None:
           return X, is_max
       return X
class Encoder(nn.Module):
   def __init__(self, aggregator, feature_dim):
       super(Encoder, self).__init__()
       self.aggregator = aggregator
       self.feature_dim = feature_dim

   def forward(self, node_features, hyper_graph):
       output = self.aggregator.forward(node_features, hyper_graph)
       return output

class HIVEMixer(nn.Module):
   def __init__(self, args, is_central=False):
       super().__init__()
       self.args = args
       self.add_self = args.add_self
       self.hyper_hidden_dim = args.hyper_hidden_dim
       self.head_num = 1
       self.hyper_edge_num = args.hyper_edge_num
       self.n_agents = args.n_agents
       self.state_dim = int(np.prod(args.state_shape))
       self.indiv_u_dim = int(np.prod(args.observation_shape))
       self.n_hyper_edge = self.hyper_edge_num
       self.is_central = is_central
       self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, 8))
       self.hyper_o_final = nn.Sequential(nn.Linear(self.indiv_u_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, 8))
       self.hyper_o_final_1 = nn.Sequential(nn.Linear(self.indiv_u_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, 8))
       self.hyper_b_final = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.hyper_hidden_dim, 8))
       self.use_elu = True
       self.hyper_edge_net_1 = nn.Sequential(
           nn.Linear(in_features=self.indiv_u_dim, out_features=self.hyper_edge_num),
       )
       self.obs_emb = nn.Sequential(
           nn.Linear(in_features=self.indiv_u_dim*2, out_features=32),
       )
       self.action_emb = nn.Sequential(
           nn.Linear(in_features=self.args.n_actions, out_features=32),
       )
       self.hyper_edge_net_1 = nn.Sequential(nn.Linear(in_features=self.indiv_u_dim,out_features=self.n_hyper_edge))
       self.hidden_dim = 64
       self.sample_size = self.args.sample_size
       self.position = Position_Game(self.n_hyper_edge,self.args.sample_size,self.state_dim, self.args.n_agents)
       self.hyper_weight_layer_1 = nn.Sequential(
           nn.Linear(self.state_dim, 3*self.hyper_hidden_dim),
           nn.ReLU(),
           nn.Linear(3*self.hyper_hidden_dim, 2*self.hyper_hidden_dim),
       )
       self.hyper_weight_layer_2 = nn.Sequential(
           nn.Linear(self.state_dim, self.hyper_hidden_dim),
           nn.ReLU(),
           nn.Linear(self.hyper_hidden_dim, self.n_agents)
       )
       self.hyper_const_layer_1 = nn.Sequential(
           nn.Linear(self.state_dim, 2 * self.hyper_hidden_dim),
           nn.ReLU(),
           nn.Linear(2 * self.hyper_hidden_dim, self.hyper_hidden_dim * self.n_agents*self.n_hyper_edge)
       )
       self.hyper_weight_layer = nn.Sequential(
           nn.Linear(self.state_dim, self.hyper_hidden_dim),
           nn.ReLU(),
           nn.Linear(self.hyper_hidden_dim, self.hyper_hidden_dim),
       )
       self.hyper_const_layer = nn.Sequential(
           nn.Linear(self.state_dim, self.hyper_hidden_dim),
           nn.ReLU(),
           nn.Linear(self.hyper_hidden_dim, self.n_agents*self.n_hyper_edge)
       )
       self.final_const_layer = nn.Sequential(
           nn.Linear(self.state_dim, self.hyper_hidden_dim),
           nn.ReLU(),
           nn.Linear(self.hyper_hidden_dim, 1)
       )
       self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.indiv_u_dim, nhead=1)
       self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
       self.hyper_edge_net = nn.Sequential(
           nn.Linear(in_features=self.indiv_u_dim, out_features=128),
           nn.ReLU(),
           nn.Dropout(0.5),
           nn.Linear(in_features=128, out_features=self.hyper_edge_num),
           nn.Dropout(0.5),
       )
   def build_hyper_net(self, indiv_us, states, actions=None, obs=None):
       bl = states.size(0)
       el = states.size(1)
       indiv_us = indiv_us.reshape(-1, indiv_us.size(-2), indiv_us.size(-1))
       out = indiv_us
       out = out.reshape(bl, el, self.n_agents, -1)#(batch_size, episode_size, n_nodes, in)
       out = out.permute(1, 2, 0, 3)#(episode_size, n_nodes , batch_size, in)
       out_emb = []
       for i in range(el):
           out_emb.append(self.transformer_encoder(out[i]))
       out_emb = torch.stack(out_emb, dim=0)#(episode_size, n_nodes , batch_size, in)
       out_emb = out_emb.permute(2, 0, 1, 3)#(batch_size, episode_size, n_nodes, in)
       out_emb = self.hyper_edge_net_1(out_emb)
       out_emb = torch.relu(out_emb)
       return out_emb.reshape(bl * el, self.n_agents, -1)

   def forward(self, agent_qs, states, indiv_us, subcoalition_map, grand_coalitions, is_max=None, actions=None, is_target=False, rest=None, use_u_max=True):
       bs = agent_qs.size(0)
       sl = agent_qs.size(1)
       agent_qs = agent_qs.view(-1, self.n_agents)
       if is_target:
           states = states.reshape(-1, states.size(-1))
           final_const = self.final_const_layer(states).squeeze(-1)
           return (agent_qs.sum(dim=-1)+final_const).view(bs, sl, 1)
       rest = rest.reshape(-1, self.n_agents, 1)
       indiv_us = indiv_us.reshape(-1, indiv_us.size(-2), indiv_us.size(-1))
       hyper_graph = self.build_hyper_net(indiv_us, states)
       states = states.reshape(-1, states.size(-1))
       final_const = self.final_const_layer(states).squeeze(-1)
       hyper_graph = hyper_graph.reshape(-1, hyper_graph.size(-2), hyper_graph.size(-1))
       node_features = agent_qs.unsqueeze(dim=-1)
       if use_u_max and (is_max is not None):
           is_max = is_max.reshape(bs*sl, self.n_agents, 1)
           is_max = is_max.unsqueeze(1).expand(-1, self.sample_size, -1, -1)
           is_max = is_max.reshape(-1, self.n_agents, 1)
           qs_tot, is_max_ = self.position(rest, hyper_graph, states, subcoalition_map, grand_coalitions, is_max)#(b, n_s, n_e, 1)
       else:
           qs_tot = self.position(rest, hyper_graph, states, subcoalition_map, grand_coalitions)#(b, n_s, n_e, 1)
       qs_tot = qs_tot.reshape(bs*sl*self.sample_size, -1, 1)
       qs_tot = qs_tot.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
       node_features = node_features.unsqueeze(1).expand(-1,self.sample_size, -1, -1)
       node_features = node_features.reshape(bs*sl*self.sample_size, -1, 1)
       node_features = node_features.unsqueeze(2).expand(-1,-1,self.n_hyper_edge, -1)
       qs_tot_ = torch.cat([qs_tot, node_features], dim=-1)
       qs_tot_ = qs_tot_.reshape(bs*sl*self.sample_size, -1, 2)
       states = states.unsqueeze(1).expand(-1, self.sample_size, -1)
       states = states.reshape(-1, self.state_dim)
       hyper_weight_1 = self.hyper_weight_layer_1(states).reshape(-1, 2, self.hyper_hidden_dim)
       hyper_const_1 = self.hyper_const_layer_1(states).reshape(-1, self.n_agents*self.n_hyper_edge, self.hyper_hidden_dim)
       q_tot = torch.bmm(qs_tot_, hyper_weight_1)/math.sqrt(2)+ hyper_const_1
       if self.use_elu:
           q_tot = F.elu(q_tot)
       hyper_weight = self.hyper_weight_layer(states).reshape(-1, self.hyper_hidden_dim, 1)
       hyper_const = self.hyper_const_layer(states).reshape(-1, self.n_agents*self.n_hyper_edge, 1)
       is_max = is_max.reshape(bs*sl, -1, 1)
       q_tot = torch.bmm(q_tot, torch.abs(hyper_weight))/math.sqrt(self.hyper_hidden_dim) + hyper_const
       q_tot = q_tot.reshape(bs*sl, self.sample_size, self.n_agents, self.n_hyper_edge)
       q_tot = torch.relu(-q_tot)
       if use_u_max and (is_max is not None):
           is_max_ = is_max_.reshape(bs*sl, self.sample_size, 1, self.n_hyper_edge)
           is_max = is_max.reshape(bs*sl, self.sample_size, self.n_agents, 1)
           is_max = torch.matmul(is_max.float(), is_max_)
           q_tot = q_tot * (1-is_max)
       q_tot = q_tot.mean(dim=1)
       q_tot = q_tot.sum(dim=-1, keepdim=True)
       if use_u_max and (is_max is not None):
           q_tot = agent_qs.sum(dim=-1) + final_const - q_tot.squeeze(-1).sum(dim=-1)
       else :
           q_tot = agent_qs.sum(dim=-1)  + final_const - (q_tot*(1 - is_max)).squeeze(-1).sum(dim=-1)
       return q_tot.view(bs, sl, 1)


