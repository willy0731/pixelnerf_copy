from torch import nn
import torch

#  import torch_scatter
import torch.autograd.profiler as profiler
import util

class ResnetBlockFC(nn.Module): # ResnetFC中的self.blocks 主要作Residual的地方
    def __init__(self, input_size, output_size=None, hidden_size=None, beta=0.0):
        super().__init__()
        if output_size is None: # default
            output_size = input_size
        if hidden_size is None: # default
            hidden_size = min(input_size, output_size)
        self.input_size = input_size # 512
        self.hidden_size = hidden_size # 512
        self.output_size = output_size # 512
        self.fc_0 = nn.Linear(input_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, output_size)
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)
        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # softplus(x)=log(1+exp(x))
        else: # default
            self.activation = nn.ReLU()
        if input_size == output_size: # default
            self.shortcut = None
        else: # 若input output size不同 則
            self.shortcut = nn.Linear(input_size, output_size, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x)) # 將x作activate後丟進Linear layer->size_h
            dx = self.fc_1(self.activation(net)) # 將net作activate後丟進Linear layer->size_out

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx

class ResnetFC(nn.Module):
    def __init__(self, d_in, d_out=4, n_blocks=5, d_latent=0, d_hidden=128,
                 beta=0.0, combine_layer=1000, combine_type="average", use_spade=False):
        super().__init__()
        if d_in > 0:
            self.linear_input = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.linear_input.bias, 0.0)
            nn.init.kaiming_normal_(self.linear_input.weight, a=0, mode="fan_in")
        
        self.linear_output = nn.Linear(d_hidden, d_out) # 轉成RGB, sigma的最後一層
        nn.init.constant_(self.linear_output.bias, 0.0)
        nn.init.kaiming_normal_(self.linear_output.weight, a=0, mode="fan_in")
        self.d_in = d_in # 6
        self.d_out = d_out # 4
        self.d_latent = d_latent # 512
        self.d_hidden = d_hidden # 512
        self.n_blocks = n_blocks # 5
        self.combine_layer = combine_layer # 3
        self.combine_type = combine_type # average
        self.use_spade = use_spade # False
        
        self.blocks = nn.ModuleList([ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)])
        if d_latent != 0: # use latent feature
            n_lin_z = min(combine_layer, n_blocks) # min(3,5)=3
            self.lin_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)])
            for i in range(n_lin_z): # 初始化這三層linear layer的weights和bias
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")
            
            if self.use_spade: # default 用不到
                self.scale_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)])
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")
        
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else: # default
            self.activation = nn.ReLU()

    def forward(self, zx, combin_inner_dims=(1,), combine_index=None, dim_size=None):
        print(zx.shape)
        print(combin_inner_dims)
        print(combine_index)
        print(dim_size)

    @classmethod    
    def from_conf(cls, conf, d_in, **kwargs):
        return cls(
                    d_in,
                    n_blocks=conf.get_int("n_blocks", 5),
                    d_hidden=conf.get_int("d_hidden", 128),
                    beta=conf.get_float("beta", 0.0),
                    combine_layer=conf.get_int("combine_layer", 1000),
                    combine_type=conf.get_string("combine_type", "average"),  # average | max
                    use_spade=conf.get_bool("use_spade", False),
                    **kwargs
        )
