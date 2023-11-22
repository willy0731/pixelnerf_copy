import torch
import numpy as np
import torch.autograd.profiler as profiler


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs # 總共有sin(x/y/z), cos(x/y/z) ~~~ sin(6x/6y/6z), cos(6x/6y/6z)
        self.d_in = d_in # x,y,z
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs) # log_sampling 
        self.d_out = self.num_freqs * 2 * d_in # 6*2*3=36
        self.include_input = include_input # 包含輸出本身
        if include_input:
            self.d_out += d_in # 36+3=39
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        # 達成同時計算sin & cos的效果
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        # 擴張為[1,12,1]
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            # 將input從[RB*B, 3]變成[RB*B, 12, 3]
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            # 將embed分別和_freqs相乘, 將結果相加於_phases上, 最後取sin
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1) # 將 12*3, 形狀變[RB*B, 36]
            if self.include_input: # 加入輸入本身
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    # 允許從配置文件中讀取相關參數，並使用這些參數創建一個PositionalEncoding類的實例
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
        )
