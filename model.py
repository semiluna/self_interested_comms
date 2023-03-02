import copy

from ray.rllib.utils import try_import_torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, SlimConv2d, SlimFC

torch, nn = try_import_torch()
from torchsummary import summary

DEFAULT_OPTIONS = {
    "activation": "relu",
    "agent_split": 1,
    "cnn_compression": 512,
    "cnn_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [128, [4, 4], 2]],
    "cnn_residual": False,
    "freeze_coop": True,
    "freeze_coop_value": False,
    "freeze_greedy": False,
    "freeze_greedy_value": False,
    "graph_edge_features": 1,
    "graph_features": 512,
    "graph_layers": 1,
    "graph_tabs": 3,
    "relative": True,
    "value_cnn_compression": 512,
    "value_cnn_filters": [[32, [8, 8], 2], [64, [4, 4], 2], [128, [4, 4], 2]],
    "forward_values": True
}

class CNN(nn.Module):
    def __init__(self, input_shape, model_config, critic=False):
        nn.Module.__init__(self)
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config['custom_model_config'])

        layers = []
        (w, h, in_channels) = input_shape
        in_size = [w, h]

        filters = self.cfg['cnn_filters'] if not critic else self.cfg['value_cnn_filters']
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            layers.append(SlimConv2d(in_channels, out_channels, kernel, stride, padding, activation_fn=self.activation))
            in_channels = out_channels
            in_size = out_size
        
        out_channels, kernel, stride =filters[-1]
        layers.append(
            SlimConv2d(in_channels, out_channels, kernel, stride, None))
        layers.append(nn.Flatten(1, -1))

        self.convolutions = nn.Sequential(*layers)

    def forward(self, map):
        agent_cnn = self.convolutions(map.permute(0, 3, 1, 2))
        return agent_cnn


class AGNN(nn.Module):
    def __init__(self, agent_id, model_config, aggregation='sum'):
        nn.Module.__init__(self)
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config['custom_model_config'])

        self.agent_id = agent_id
        self.graph_features = self.cfg['graph_features']
        self.hops = self.cfg['graph_tabs']

        self.layers = []

        for _ in range(self.cfg['graph_layers']):
            layer_filters = nn.ModuleList()
            for _ in range(self.hops):
                layer_filters.append(nn.Parameter(self.graph_features, self.graph_features))
            self.layers.append(layer_filters)
        
        self.aggregation = {
            "sum": lambda y, dim: torch.sum(y, dim=dim),
            "median": lambda y, dim: torch.median(y, dim=dim)[0],
            "min": lambda y, dim: torch.min(y, dim=dim)[0]
        }[aggregation]

        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]
    
    # NOTE: I am performing a naive forward approach for batched inputs...
    # I may re-implement this to be faster some other time. 
    def forward(self, batched_x, batched_adj_mat):
        bs, features = batched_x.shape[0], batched_x.shape[1]
        output = torch.zeros(bs, features)
        for b in range(bs):
            x = batched_x[b]
            adj_mat = batched_adj_mat[b]
            for l in range(self.layers):
                out = torch.linalg.matrix_power(adj_mat, 0)[self.agent_id, :] @ x @ self.layers[l][h]
                for h in range(1, self.hops):
                    Sk = torch.linalg.matrix_power(adj_mat, h)[self.agent_id, :]
                    out += Sk @ x @ self.layers[l][h]
                
                out = self.activation(out)
            output[b] = out
        return output

class MLP(nn.Module):
    def __init__(self, dims, cfg):
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(cfg['custom_model_config'])

        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

        input_dim, hidden_dim_1, hidden_dim_2, out_dim = dims
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            self.activation(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            self.activation(),
            nn.Linear(hidden_dim_2, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class AdversarialModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config['custom_model_config'])

        self.n_agents = self.cfg['n_agents']
        # ===================== ACTORS =======================
        self.agent_cnns = [CNN(obs_space.original_space['agents'][a_id]['map'].shape, model_config) for a_id in range(self.n_agents)]
        self.agent_gnns = [AGNN(a_id, model_config) for a_id in range(len(self.n_agents))]

        logit_feats = self.graph_features
        if self.cfg['cnn_residual']:
            logit_feats += self.cnn_compression
        
        self.agent_mlps = [MLP([logit_feats, 64, 32, 5], model_config) for _ in range(len(self.n_agents))]

        # ===================== CRITICS =======================
        self.global_cnns = [CNN(obs_space.original_space['state'].shape, model_config, critic=True) for _ in range(self.n_agents)]
        self.critic_cnns = [CNN(obs_space.original_space['agents'][a_id]['map'].shape, model_config) for a_id in range(self.n_agents)]
        self.critic_mlps = [MLP([self.cfg['cnn_compression'] + self.cfg['value_cnn_compression'], 64, 32, 1], model_config) for _ in range(self.n_agents)]

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict['obs']['gso'].shape[0]
        o_as = input_dict['obs']['agents']

        # ===================== ACTORS ========================
        gsos = input_dict['obs']['gso']

        logits = torch.empty(batch_size, self.n_agents, 5)
        for a_id in range(self.n_agents):
            cnn_feats = self.agent_cnns[a_id](o_as[a_id]['map'])
            gnn_feats = self.agent_gnns[a_id](cnn_feats, gsos)
            
            if self.cfg['cnn_residual']:
                mlp_input = torch.cat([gnn_feats, cnn_feats], dim=1)
            else:
                mlp_input = gnn_feats
            
            logits[:, a_id] = self.agent_mlps[a_id](mlp_input)

        # ===================== CRITICS =====================
        values = torch.empty(batch_size, self.n_agents)
        if self.cfg['forward_values']:
            for a_id in range(self.n_agents):
                local_feats = self.critic_cnns[a_id](o_as[a_id]['map'])
                global_feats = self.global_cnns[a_id](input_dict['obs']['state'])
                total_feats = torch.cat([local_feats, global_feats], dim=1)
                values[:, a_id] = self.critic_mlps[a_id](total_feats).squeeze(1)
        
        self._cur_value = values

        return logits.view(batch_size, self.n_agents * 5), state
    
    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value




        
        