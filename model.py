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
    "forward_values": True,
    "disable_comms": False,
}

class CNN(nn.Module):
    def __init__(self, input_shape, model_config, critic=False):
        nn.Module.__init__(self)
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config)

        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

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
        self.cfg.update(model_config)

        self.agent_id = agent_id
        self.graph_features = self.cfg['graph_features']
        self.hops = self.cfg['graph_tabs']

        self.layers = nn.ParameterList()

        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

        for _ in range(self.cfg['graph_layers']):
            layer_filters = nn.ParameterList()
            for _ in range(self.hops):
                aux = torch.empty(self.graph_features, self.graph_features)
                nn.init.uniform_(aux)
                layer_filters.append(nn.parameter.Parameter(aux))
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
        bs, n_agents, features = batched_x.shape
        assert batched_adj_mat.shape == (bs, n_agents, n_agents)

        output = torch.zeros(bs, features)
        for b in range(bs):
            x = batched_x[b]
            adj_mat = batched_adj_mat[b]
            for layer in self.layers:
                out = torch.linalg.matrix_power(adj_mat, 0)[self.agent_id, :] @ x @ layer[0]
                for h in range(1, self.hops):
                    Sk = torch.linalg.matrix_power(adj_mat, h)[self.agent_id, :]
                    out += Sk @ x @ layer[h]
                
                out = self.activation()(out)
            output[b] = out
        return output

class MLP(nn.Module):
    def __init__(self, dims, cfg):
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(cfg)

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
    def __init__(self, obs_space, action_space, num_outputs, config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, config, name)
        nn.Module.__init__(self)

        model_config = config['custom_model_config']
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config)
        self.n_agents = len(action_space)


        self.disable_comms = self.cfg['disable_comms']

        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

        self.cnn_out = self.cfg['cnn_compression']
        # ===================== ACTORS =======================
        self.agent_cnns = [CNN(obs_space.original_space['agents'][a_id]['map'].shape, model_config) for a_id in range(self.n_agents)]
        self.agent_gnns = [AGNN(a_id, model_config) for a_id in range(self.n_agents)]

        self.agent_cnns = nn.ModuleList(self.agent_cnns)
        self.agent_gnns = nn.ModuleList(self.agent_gnns)

        logit_feats = self.cfg['graph_features']
        if self.cfg['cnn_residual']:
            logit_feats += self.cnn_out
        
        self.agent_mlps = nn.ModuleList([MLP([logit_feats, 64, 32, 5], model_config) for _ in range(self.n_agents)])

        # ===================== CRITICS =======================
        self.cnn_value_out = self.cfg['value_cnn_compression']
        self.global_cnns = nn.ModuleList([CNN(obs_space.original_space['state'].shape, model_config, critic=True) for _ in range(self.n_agents)])
        self.critic_cnns = nn.ModuleList([CNN(obs_space.original_space['agents'][a_id]['map'].shape, model_config) for a_id in range(self.n_agents)])
        self.critic_mlps = nn.ModuleList([MLP([self.cnn_out + self.cnn_value_out, 64, 32, 1], model_config) for _ in range(self.n_agents)])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict['obs']['gso'].shape[0]
        o_as = input_dict['obs']['agents']

        # ===================== ACTORS ========================
        device = input_dict['obs']['gso'].device
        gsos = input_dict['obs']['gso'].to(device)
        if self.disable_comms:
            aux = torch.eye(self.n_agents)
            aux = aux.reshape((1, self.n_agents, self.n_agents))
            batched = aux.repeat(batch_size, 1, 1)
            gsos = batched.to(device) # the adjancency matrix has only self loops

        device = gsos.device

        logits = torch.empty(batch_size, self.n_agents, 5).to(device)
        # get features from CNN compression
        feats = []
        for a_id in range(self.n_agents):
            feats.append(self.agent_cnns[a_id](o_as[a_id]['map']))
        cnn_feats = torch.stack(feats, dim=1).to(device)

        # perform message passing on these features
        feats = []
        for a_id in range(self.n_agents):
            feats.append(self.agent_gnns[a_id](cnn_feats, gsos))
        gnn_feats = torch.stack(feats, dim=1).to(device)

        if self.cfg['cnn_residual']:
            mlp_input = torch.cat([gnn_feats, cnn_feats], dim=-1).to(device)
        else:
            mlp_input = gnn_feats.to(device)
        
        # compress final GNN features per node to actions
        for a_id in range(self.n_agents):    
            logits[:, a_id] = self.agent_mlps[a_id](mlp_input[:, a_id])

        # ===================== CRITICS =====================
        values = torch.empty(batch_size, self.n_agents).to(device)
        if self.cfg['forward_values']:
            for a_id in range(self.n_agents):
                local_feats = self.critic_cnns[a_id](o_as[a_id]['map'])
                global_feats = self.global_cnns[a_id](input_dict['obs']['state'])
                total_feats = torch.cat([local_feats[:, :self.cnn_out], 
                                        global_feats[:, :self.cnn_value_out]], dim=1)
                
                values[:, a_id] = self.critic_mlps[a_id](total_feats).squeeze(1)
        
        self._cur_value = values

        return logits.view(batch_size, self.n_agents * 5), state
    
    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value




        
        