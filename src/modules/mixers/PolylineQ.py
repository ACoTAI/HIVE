import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """使用交叉注意力增强多智能体交互"""

    def __init__(self, state_shape, n_agents, n_actions, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        # 状态嵌入
        self.state_embed = nn.Linear(state_shape, embed_dim // 2)

        # 动作嵌入（每个智能体的动作分开处理）
        self.action_embed = nn.Linear(n_actions, embed_dim // 2)

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 输出层
        self.output_layer = nn.Linear(embed_dim + 1, 1)


    def forward(self, x, q_vals):
        batch_size = x.shape[0]

        # 分离状态和动作
        state = x[:, :self.state_shape]
        actions = x[:, self.state_shape:]
        actions = actions.view(batch_size, self.n_agents, -1)

        # 嵌入状态和动作
        state_embed = self.state_embed(state).unsqueeze(1).expand(-1, self.n_agents, -1)
        action_embed = self.action_embed(actions)

        # 组合状态和动作嵌入
        agent_embeddings = torch.cat([state_embed, action_embed], dim=-1)  # [batch, n_agents, embed_dim]

        # 交叉注意力
        attn_output, _ = self.cross_attention(agent_embeddings, agent_embeddings, agent_embeddings)

        # 残差连接
        agent_embeddings = self.norm1(agent_embeddings + attn_output)

        # 前馈网络
        ffn_output = self.ffn(agent_embeddings)
        agent_embeddings = self.norm2(agent_embeddings + ffn_output)

        q_vals = q_vals.reshape(-1, self.n_agents, 1)
        agent_embeddings = torch.cat([agent_embeddings, q_vals], dim=-1)

        # 输出每个智能体的值
        outputs = self.output_layer(agent_embeddings).squeeze(-1)  # [batch, n_agents]

        return outputs


class PolylineQ(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.state_dim = args.state_shape
        self.b =nn.Sequential(
            nn.Linear(args.state_shape, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.w_1 = nn.Sequential(
            nn.Linear(args.state_shape, 64),
            nn.GELU(),
            nn.Linear(64, self.args.n_agents),
        )
        self.b_1 = nn.Sequential(
            nn.Linear(args.state_shape, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        if args.with_atten:
            self.w = CrossAttention(args.state_shape, args.n_agents, args.n_actions)
        else:
            self.w = nn.Sequential(
                nn.Linear(args.state_shape + args.n_agents*args.n_actions, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, args.n_agents)
            )
        self.n_actions = args.n_actions

    def forward(self, states, q_vals, obs_act, is_max_action):
        b = self.b(states)
        w_1 = torch.abs(self.w_1(states))
        b_1 = self.b_1(states)
        q_vals = w_1 * q_vals + b_1
        if self.args.with_atten:
            act = obs_act[:, :, :, -self.n_actions:].reshape(states.shape[0], states.shape[1], -1)
            w = self.w(torch.cat([states, act], dim=-1).view(-1, self.args.state_shape + self.args.n_agents*self.args.n_actions), q_vals)
            w = torch.abs(w) + 1
            w = w.reshape(states.shape[0], states.shape[1], self.args.n_agents)
            w = w.unsqueeze(-1)
        else:
            act = obs_act[:,:,:, -self.n_actions:].reshape(states.shape[0], states.shape[1], -1)
            w = self.w(torch.cat([states, act], dim=-1))
            w = torch.abs(w) + 1
            w = w.unsqueeze(-1)
        w = is_max_action + (1 - is_max_action)*w
        w = w.squeeze(-1)
        is_pos = (q_vals > 0).float()
        return (q_vals * (is_pos + (1 - is_pos)*w)).sum(dim=-1, keepdim=True) + b
