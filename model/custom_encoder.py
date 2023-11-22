import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class ConvEncoder(nn.Module):
    """
    Basic, extremely simple convolutional encoder
    把input分別丟進三個convolution layer
    在兩個間隔中,分別插入三個conv,和三個deconv 來完成encode和decode
    """

    def __init__(self, dim_in=3, norm_layer=util.get_norm_layer("group"), padding_type="reflect", use_leaky_relu=True, use_skip_conn=True):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.padding_type = padding_type
        self.use_skip_conn = use_skip_conn

        # TODO: make these configurable
        first_layer_chnls = 64 
        mid_layer_chnls = 128
        last_layer_chnls = 128
        n_down_layers = 3
        self.n_down_layers = n_down_layers

        self.conv_in = nn.Sequential( # input convolution layer
            nn.Conv2d(dim_in, first_layer_chnls, kernel_size=7, stride=2, bias=False),
            norm_layer(first_layer_chnls), # 64
            self.activation,
        )

        chnls = first_layer_chnls # 64
        for i in range(0, n_down_layers): # 3個conv block & deconv block
            conv = nn.Sequential( # (64,128), (128,256), (256,512)
                nn.Conv2d(chnls, 2 * chnls, kernel_size=3, stride=2, bias=False), # 128
                norm_layer(2 * chnls), # 128, 256, 512
                self.activation,
            )
            setattr(self, "conv" + str(i), conv)

            deconv = nn.Sequential(
                nn.ConvTranspose2d( # Upsampling 反卷積
                    4 * chnls, chnls, kernel_size=3, stride=2, bias=False
                ), # (256,64), (512,128), (1024,256)
                norm_layer(chnls), # 64, 128, 256
                self.activation,
            )
            setattr(self, "deconv" + str(i), deconv)
            chnls *= 2

        self.conv_mid = nn.Sequential( # middle convolution layer
            nn.Conv2d(chnls, mid_layer_chnls, kernel_size=4, stride=4, bias=False),
            norm_layer(mid_layer_chnls),
            self.activation,
        )

        self.deconv_last = nn.ConvTranspose2d( # last convolution layer # 反卷積層
            first_layer_chnls, last_layer_chnls, kernel_size=3, stride=2, bias=True
        )

        self.dims = [last_layer_chnls]

    def forward(self, x):
        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_in)
        x = self.conv_in(x) # 丟入初始層

        inters = []
        # 用"same_pad_conv2d"確保輸出大小相同 並丟進conv_i運算三次
        for i in range(0, self.n_down_layers):
            conv_i = getattr(self, "conv" + str(i))
            x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=conv_i)
            x = conv_i(x)
            inters.append(x)

        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_mid)
        x = self.conv_mid(x) # 丟入中間層
        # x.shape = [Batch, channels, H, W]
        # 將x的H,W改成最後一個inters中feature map的H,W
        x = x.reshape(x.shape[0], -1, 1, 1).expand(-1, -1, *inters[-1].shape[-2:])

        # 用"same_unpad_conv2d"確保輸出大小相同 並丟進deconv_i運算三次
        for i in reversed(range(0, self.n_down_layers)): # 反向迴圈, 2 -> 1 -> 0
            if self.use_skip_conn: # 若skip connection使用, 將feature map與x合併
                x = torch.cat((x, inters[i]), dim=1)
            deconv_i = getattr(self, "deconv" + str(i))
            x = deconv_i(x)
            # 用"same_unpad_conv2d"將x重塑成符合下一層輸入的形狀
            x = util.same_unpad_deconv2d(x, layer=deconv_i)
        x = self.deconv_last(x) # 丟入最後一層
        x = util.same_unpad_deconv2d(x, layer=self.deconv_last)
        return x
