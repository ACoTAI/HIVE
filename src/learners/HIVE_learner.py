import copy
import torch
from components.episode_buffer import EpisodeBatch
from modules.mixers.HIVE import HIVEMixer
import torch as th
from torch.optim import RMSprop, Adam


class HGCNLeaner_new:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = self.args.n_agents
        self.n_edges = 64

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = HIVEMixer(self.args)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        self.rest = copy.deepcopy(self.mac)
        #self.rest._build_agents(self.rest._get_input_shape(scheme), hetero=True)
        self.params += list(self.rest.parameters())
        self.optimiser = Adam(params=self.params, lr=args.lr)
        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.sample_size = args.sample_size
        self.n_ = args.hyper_edge_num
        self.need_print = (self.args.n_agents == 2)
    def sample_grandcoalitions(self, batch_size):
        cuda_ = "cuda" if self.args.use_cuda else "cpu"
        seq_set = th.tril(th.ones(self.n_, self.n_, device=cuda_), diagonal=0, out=None)
        grand_coalitions_pos = th.multinomial(th.ones(batch_size*self.sample_size, self.n_,device=cuda_)/self.n_, self.n_, replacement=False) # shape = (b*n_s, n)
        individual_map = th.zeros(batch_size*self.sample_size*self.n_, self.n_,device=cuda_)
        individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)#独热编码
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
        subcoalition_map = th.matmul(individual_map, seq_set)

        # FIX: construct torche grand coalition (in sequence by agent_idx) from torche grand_coalitions_pos (e.g., pos_idx <- grand_coalitions_pos[agent_idx])
        offset = (th.arange(batch_size*self.sample_size,device=cuda_)*self.n_).reshape(-1, 1)# shape = (b*n_s, 1)
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions = th.zeros_like(grand_coalitions_pos_alter.flatten(),device=cuda_)
        grand_coalitions[grand_coalitions_pos_alter.flatten()] = th.arange(batch_size*self.sample_size*self.n_,device=cuda_)
        grand_coalitions = grand_coalitions.reshape(batch_size*self.sample_size, self.n_) - offset
        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size,
            self.n_, self.n_).contiguous().view(batch_size, self.sample_size, self.n_, self.n_) # shape = (b, n_s, n, n)

        return subcoalition_map, grand_coalitions
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, probs=None, priori=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        self.need_print = self.need_print and t_env - self.log_stats_t >= self.args.learner_log_interval
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        if "matrix" in self.args.name and self.need_print:
            actions_ = []
            for i in range(self.args.n_agents):
                action = []
                for j in range(self.args.n_actions):
                    action.append(j)
                actions_.append(action)
            actions_ = torch.tensor(actions_).unsqueeze(0).unsqueeze(0).expand(actions.size(0), actions.size(1), -1, -1)
            actions_ = actions_.to(actions.device)
            demo_action_qval = th.gather(mac_out[:, :-1], dim=3, index=actions_)
            demo_q = demo_action_qval.mean(dim=0).mean(dim=0)

        rest_mac_out = []
        self.rest.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.rest.forward(batch, t=t)
            rest_mac_out.append(agent_outs)
        rest_mac_out = th.stack(rest_mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        if "matrix" in self.args.name and self.need_print:
            actions_ = []
            for i in range(self.args.n_agents):
                action = []
                for j in range(self.args.n_actions):
                    action.append(j)
                actions_.append(action)
            actions_ = torch.tensor(actions_).unsqueeze(0).unsqueeze(0).expand(actions.size(0), actions.size(1), -1, -1)
            actions_ = actions_.to(actions.device)
            demo_rest_qval = th.gather(rest_mac_out[:, :-1], dim=3, index=actions_)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        target_mac_out[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out[:, 1:], 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        mac_out_detach_ = mac_out.clone().detach()
        mac_out_detach_[avail_actions == 0] = -9999999
        cur_max_actions = mac_out_detach_[:, :-1].max(dim=3, keepdim=True)[1]
        #is_max_action = (actions.detach() == cur_max_actions).min(dim=2)[0].long()
        is_max_action = (actions.detach() == cur_max_actions).float()
        # Mix
        if self.mixer is not None:
            bl = rewards.shape[0]
            el = rewards.shape[1]
            subcoalition_map, grand_coalitions = self.sample_grandcoalitions(bl*el)
            tot_chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch['obs'][:, :-1], subcoalition_map, grand_coalitions, is_max=is_max_action, is_target=False, rest=rest_chosen_action_qvals)
            # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch['obs'][:, :-1], None)
            tot_target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch['obs'][:, 1:], subcoalition_map, grand_coalitions, is_target=True)
            # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch['obs'][:, 1:], None)
        # Calculate 1-step Q-Learning targets

        if "matrix" in self.args.name and self.need_print:
            tot_qs = []
            for i in range(self.args.n_actions ** self.args.n_agents):
                demo_action_qvals = []
                demo_rest_qvals = []
                t = i
                idx = 0
                cur_actions = []
                while idx<self.args.n_agents:
                    demo_action_qvals.append(demo_action_qval[:, :, idx, t % self.args.n_actions].unsqueeze(-1))
                    demo_rest_qvals.append(demo_rest_qval[:, :, idx, t % self.args.n_actions].unsqueeze(-1))
                    cur_actions.append(th.ones(actions.shape[:2] + (1,)).to(actions.device) * (t % self.args.n_actions))
                    t = t // self.args.n_actions
                    idx += 1
                demo_rest_qvals = th.cat(demo_rest_qvals, dim=2)
                demo_action_qvals = th.cat(demo_action_qvals, dim=2)
                cur_actions = th.cat(cur_actions, dim=2)
                demo_is_max = (cur_actions.unsqueeze(-1) == cur_max_actions).min(dim=2)[0].long()
                tot_q = self.mixer(demo_action_qvals, batch["state"][:, :-1], batch['obs'][:, :-1], subcoalition_map, grand_coalitions, is_max=demo_is_max, is_target=False, rest=demo_rest_qvals)
                tot_qs.append(tot_q.mean(dim=0).mean(dim=0))

        targets = rewards + self.args.gamma * (1 - terminated) * tot_target_max_qvals

        # Td-error
        td_error = (tot_chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        #loss += 1e-2*((rest*mask)**2).sum()/(mask.sum()*self.n_agents)
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env
            if "matrix" in self.args.name and self.need_print:
                print_cnxt = ""
                for i in range(self.args.n_agents):
                    p_array = []
                    for j in range(demo_q[i].shape[0]):
                        p_array.append(demo_q[i][j].item())
                    print_cnxt += "q_value of agent {} 's: {}".format(i, p_array)
                    print_cnxt += '\n'
                for i in range(self.args.n_actions):
                    for j in range(self.args.n_actions):
                        print_cnxt += (str(tot_qs[j*self.args.n_actions + i].item()) + ' ')
                    print_cnxt += '\n'
                #print(print_cnxt)
                with open('{}.txt'.format(self.args.name),'w', encoding='utf-8') as file:
                    file.write(print_cnxt)
        self.need_print = (self.args.n_agents == 2)
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.rest.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
