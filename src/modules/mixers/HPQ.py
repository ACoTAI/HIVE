import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np
import math
class HGCNEncoder(nn.Module):
    def __init__(self, n_edges, state_dim=1):
        super().__init__()
        print(n_edges)
        #self.W_line = nn.Parameter(torch.ones(n_edges))
        self.W_line = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, n_edges),
        ).cuda()
        self.state_dim = state_dim
        self.W = None
        self.n_ = n_edges
    def forward(self, node_features, hyper_graph, states):
        #self.W = torch.diag_embed(self.W_line)#超边的权重（e_nodes,e_nodes)
        self.W = torch.diag_embed(self.W_line(states.reshape(-1, self.state_dim)))  # 超边的权重（n_e,n_e)
        self.W = self.W.reshape(-1, self.n_, self.n_)
        B_inv = torch.sum(hyper_graph.detach(), dim=-2)#得到每条边对应结点的个数（batch_size, 1, e_nodes）
        B_inv = torch.diag_embed(B_inv)#每条边对应结点数的对角矩阵（batch_size, e_nodes, e_nodes）
        softmax_w = torch.abs(self.W).detach()
        D_inv = torch.matmul(hyper_graph.detach(), softmax_w).sum(dim=-1)#加权度矩阵(batch_size, n_nodes, 1)
        D_inv = torch.diag_embed(D_inv)#每条边对应节点数加权的对角矩阵(batch_size, n_nodes, n_nodes)
        D_inv = D_inv **(-0.5)
        B_inv = B_inv **(-1)
        D_inv[D_inv == float('inf')] = 0
        D_inv[D_inv == float('nan')] = 0
        B_inv[B_inv == float('inf')] = 0
        B_inv[B_inv == float('nan')] = 0
        A = torch.bmm(D_inv, hyper_graph)#(batch_size, n_nodes, e_nodes)
        A = torch.matmul(A, torch.abs(self.W))#(batch_size, n_nodes, e_nodes)
        #A = torch.bmm(A, B_inv)#(batch_size, n_nodes, e_nodes)
        # A = torch.bmm(A, hyper_graph.transpose(-2, -1))#(batch_size, n_nodes, n_nodes)
        # A = torch.bmm(A, D_inv)#(batch_size, n_nodes, n_nodes)
        X = torch.bmm(A.transpose(-2, -1), node_features)#(batch_size, n_nodes, 1)
        X = F.leaky_relu(X, negative_slope=0.1)
        return X



class HPQMixer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.indiv_u_dim = int(np.prod(args.observation_shape))
        self.embed_dim = args.mixing_embed_dim
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.sample_size = args.sample_size
        self.too_many = (self.n_agents > 4)
        if self.too_many:
            self.hyper_edge_num = args.hyper_edge_num
        else:
            self.hyper_edge_num = args.hyper_edge_num + args.n_agents
        self.n_hyper_edge = args.hyper_edge_num
        self.n_ = self.hyper_edge_num
        self.hyper_hidden_dim = args.hyper_hidden_dim
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * 2)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.GELU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * 2))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.GELU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        # State dependent bias for the last layers
        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                       nn.GELU(),
                                       nn.Linear(self.embed_dim, 1))
        self.Encoder = HGCNEncoder(self.hyper_edge_num, state_dim=self.state_dim)
        self.depth = 2
        self.hid_K = nn.ModuleList([
            nn.Linear(self.indiv_u_dim, self.indiv_u_dim) for _ in range(self.depth)])
        self.hid_Q = nn.ModuleList([
            nn.Linear(self.indiv_u_dim, self.indiv_u_dim)for _ in range(self.depth)])
        self.hid_V = nn.ModuleList([
            nn.Linear(self.indiv_u_dim, self.indiv_u_dim) for _ in range(self.depth)])
        self.hyper_net = nn.Sequential(
            nn.Linear(self.indiv_u_dim, self.hyper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.hyper_hidden_dim, self.n_hyper_edge)
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(self.indiv_u_dim) for _ in range(self.depth)])

        self.hyper_b = nn.Sequential(
            nn.Linear(self.state_dim, 2*self.state_dim),
            nn.GELU(),
            nn.Linear(2*self.state_dim, 1)
        )
    def sample_grandcoalitions(self, batch_size):
        """
        E.g. batch_size = 2, n_agents = 3:

        >>> grand_coalitions_pos
        tensor([[2, 0, 1],
                [1, 2, 0]])

        >>> subcoalition_map
        tensor([[[[1., 1., 1.],
                [1., 0., 0.],
                [1., 1., 0.]]],

                [[[1., 1., 0.],
                [1., 1., 1.],
                [1., 0., 0.]]]])

        >>> individual_map
        tensor([[[[0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.]]],

                [[[0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]]]])
        """
        seq_set = th.tril(th.ones(self.n_, self.n_).cuda(), diagonal=0, out=None)
        grand_coalitions_pos = th.multinomial(th.ones(batch_size * self.sample_size,
                                                      self.n_).cuda() / self.n_,
                                              self.n_,
                                              replacement=False)
        individual_map = th.zeros(batch_size * self.sample_size * self.n_, self.n_).cuda()
        individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
        subcoalition_map = th.matmul(individual_map, seq_set)

        # FIX: construct the grand coalition (in sequence by agent_idx) from the grand_coalitions_pos (e.g., pos_idx <- grand_coalitions_pos[agent_idx])
        offset = (th.arange(batch_size * self.sample_size) * self.n_).reshape(-1, 1).cuda()
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions = th.zeros_like(grand_coalitions_pos_alter.flatten()).cuda()
        grand_coalitions[grand_coalitions_pos_alter.flatten()] = th.arange(
            batch_size * self.sample_size * self.n_).cuda()
        grand_coalitions = grand_coalitions.reshape(batch_size * self.sample_size, self.n_) - offset

        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size * self.sample_size,
                                                                self.n_,
                                                                self.n_).contiguous().view(batch_size,
                                                                                                 self.sample_size,
                                                                                                 self.n_,
                                                                                                 self.n_)  # shape = (b, n_s, n, n)
        return subcoalition_map, individual_map, grand_coalitions

    def get_beta_estimate(self, states, agent_qs, res=None):
        batch_size = states.size(0)

        # get subcoalition map including agent i
        subcoalition_map, individual_map, grand_coalitions = self.sample_grandcoalitions(
            batch_size)  # shape = (b, n_s, n, n)

        # reshape the grand coalition map for rearranging the sequence of actions of agents
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size,
                                                                 self.sample_size,
                                                                 self.n_,
                                                                 self.n_,
                                                                 1)  # shape = (b, n_s, n, n, 1)

        # remove agent i from the subcloation map
        subcoalition_map_no_i = subcoalition_map - individual_map
        subcoalition_map_no_i = subcoalition_map_no_i.unsqueeze(-1).expand(batch_size,
                                                                           self.sample_size,
                                                                           self.n_,
                                                                           self.n_,
                                                                           1)  # shape = (b, n_s, n, n, 1)

        # reshape actions for further process on coalitions
        if res is None:
            reshape_agent_qs = agent_qs.unsqueeze(1).unsqueeze(2).expand(batch_size,
                                                                     self.sample_size,
                                                                     self.n_,
                                                                     self.n_,
                                                                     1).gather(3,
                                                                               grand_coalitions)  # shape = (b, n, 1) -> (b, 1, 1, n, 1) -> (b, n_s, n, n, 1)
        else:
            reshape_agent_qs = res.unsqueeze(1).unsqueeze(2).expand(batch_size,
                                                                         self.sample_size,
                                                                         self.n_,
                                                                         self.n_,
                                                                         1).gather(3,
                                                                                   grand_coalitions)
        # get actions of its coalition memebers for each agent
        agent_qs_coalition = reshape_agent_qs * subcoalition_map_no_i  # shape = (b, n_s, n, n, 1)

        # get actions vector of its coalition members for each agent
        subcoalition_map_no_i_ = subcoalition_map_no_i.sum(dim=-2).clone()
        subcoalition_map_no_i_[subcoalition_map_no_i.sum(dim=-2) == 0] = 1
        agent_qs_coalition_norm_vec = agent_qs_coalition.sum(dim=-2) / subcoalition_map_no_i_  # shape = (b, n_s, n, 1)

        reshape_agent_qs_coalition_norm_vec = agent_qs_coalition_norm_vec.mean(dim=1).reshape(-1, 1)
        reshape_agent_qs_individual = agent_qs.view(-1, 1)
        reshape_states = states.unsqueeze(1).expand(-1, self.n_, -1).contiguous().view(-1, self.state_dim)
        inputs = th.cat([reshape_agent_qs_coalition_norm_vec, reshape_agent_qs_individual], dim=-1).unsqueeze(
            1)  # shape = (b*n_s*n, 1, 2*1)

        # First layer
        w1 = self.hyper_w_1(reshape_states)
        b1 = self.hyper_b_1(reshape_states)
        w1 = w1.view(-1, 2, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.gelu(th.bmm(inputs, w1) + b1)
        # Second layer
        w_final = self.hyper_w_final(reshape_states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        b2 = self.hyper_b_2(reshape_states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + b2
        # Reshape and return
        if self.args.use_abs:
            beta_estimates = th.abs(y).view(batch_size, self.n_)  # shape = (b, n)
        else:
            beta_estimates = th.sqrt(self.args.alpha * y + 1) - 1

        return beta_estimates

    def build_hyper_net(self, indiv_us):
        indiv_us = indiv_us.reshape(-1, self.n_agents , self.indiv_u_dim)
        for i in range(self.depth):
            res = indiv_us
            K = self.hid_K[i](indiv_us)#(batch_size, n_agents, hyper_hidden_dim)
            Q = self.hid_Q[i](indiv_us)#(batch_size, n_agents, hyper_hidden_dim)
            out = torch.matmul(K, Q.permute(0, 2, 1))#(batch_size, n_agents, n_agents)
            out = torch.softmax(out/math.sqrt(self.hyper_hidden_dim), dim=-1)
            indiv_us = self.norm_layers[i](torch.bmm(out, self.hid_V[i](indiv_us)) + res)#(batch_size, n_agents, hidden_dim)
        out = self.hyper_net(indiv_us)#(batch_size, n_agents, n_hyper_edge)
        out = torch.relu(out)
        if not self.too_many:
            out = torch.cat([out, torch.eye(self.n_agents).to(out.device).unsqueeze(0).expand(out.size(0), -1, -1)],dim=-1)
        return out

    def forward(self, agent_qs, states, indiv_us, max_filter, target, manual_beta_estimates=None, feat=None):
        # agent_qs, max_filter = (b, t, n)
        bs = agent_qs.size(0)
        sl = agent_qs.size(1)
        agent_qs = agent_qs.view(bs * sl, -1)
        b = self.hyper_b(states)
        reshape_states = states.contiguous().view(-1, self.state_dim)
        indiv_us = indiv_us.reshape(-1, indiv_us.size(-2), indiv_us.size(-1))
        hyper_graph = self.build_hyper_net(indiv_us)
        agent_qs = self.Encoder(agent_qs.unsqueeze(-1), hyper_graph, states)
        reshape_agent_qs = agent_qs.unsqueeze(-1).contiguous().view(-1, self.n_, 1)
        agent_qs = agent_qs.view(bs, sl, -1)
        if target:
            return th.sum(agent_qs, dim=2, keepdim=True)+b
        else:
            if manual_beta_estimates == None:
                if feat is not None:
                    feat = feat.view(bs * sl, -1)
                    if not self.too_many:
                        feat = self.Encoder(feat.unsqueeze(-1), hyper_graph, states)
                        reshape_feat = feat.unsqueeze(-1).contiguous().view(-1, self.n_, 1)
                    else:
                        reshape_feat = reshape_agent_qs
                else:
                    reshape_feat = None
                beta_estimates = self.get_beta_estimate(reshape_states, reshape_agent_qs, reshape_feat)
                beta_estimates = beta_estimates + 1
                beta_estimates = beta_estimates.contiguous().view(states.size(0), states.size(1), self.n_)
            else:
                beta_estimates = manual_beta_estimates * th.ones_like(max_filter)
            # agent with non-max action will be given 1
            max_filter = max_filter.detach()
            max_filter = max_filter.reshape(bs * sl, -1, 1)
            max_filter = ((hyper_graph * max_filter).sum(dim=1) == hyper_graph.sum(dim=1)).float().reshape(bs, sl, -1)
            non_max_filter = 1 - max_filter
            beta_estimates = (agent_qs > 0).detach().float() * 1 + (agent_qs <= 0).detach().float() * beta_estimates
            # the Q_jt was described as in Eq.(32)
            return ((beta_estimates * non_max_filter + max_filter) * agent_qs).sum(dim=2,
                                                                                    keepdim=True)+b