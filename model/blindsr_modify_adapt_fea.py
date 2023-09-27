import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder import MoCo
import functools

def make_model(args):
    return BlindSR(args)


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        
        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))
        
        # branch 2
        out = out + self.ca(x)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        #self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = conv(n_feat*2, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation kernel representation: B * C
        :param x[2]: degradation noise representation: B * C
        '''
        #print(x[0].shape)
        #print(x[1].shape)
        #print(x[2].shape)
        x_in = [x[0], x[1]]
        out = self.relu(self.da_conv1(x_in))
        out = self.relu(self.conv1(out))
        noise = torch.unsqueeze(x[2],-1)
        noise = torch.unsqueeze(noise,-1)
        noise = noise.repeat(1,1,out.shape[2],out.shape[3])
        out = torch.cat((out,noise),dim=1)
        out = self.relu(self.da_conv2(out))
        #out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]
        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        x[1] = x[1].squeeze(1)
        x[2] = x[2].squeeze(1)
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1], x[2]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class DASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DASR, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(args.scale[0])

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress_kernel = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )
        # compress
        self.compress_noise = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )
        self.k_adapt_1 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.k_adapt_2 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.k_adapt_3 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.k_adapt_4 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.k_adapt_5 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        
        self.n_adapt_1 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.n_adapt_2 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.n_adapt_3 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.n_adapt_4 = nn.Conv1d(1, 1, kernel_size=3,padding=1)
        self.n_adapt_5 = nn.Conv1d(1, 1, kernel_size=3,padding=1)        
        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v_kernel, k_v_noise):
        kernel_fea_list = []
        noise_fea_list = []
        k_v_kernel = self.compress_kernel(k_v_kernel)
        k_v_noise = self.compress_noise(k_v_noise)
        #adapt
        k_v_kernel = k_v_kernel.unsqueeze(1)
        k_v_noise = k_v_noise.unsqueeze(1)
        k_v_kernel = self.k_adapt_1(k_v_kernel)
        kernel_fea_list.append(k_v_kernel)
        k_v_noise = self.n_adapt_1(k_v_noise)
        noise_fea_list.append(k_v_noise)
        k_v_kernel = self.k_adapt_2(k_v_kernel)
        kernel_fea_list.append(k_v_kernel)
        k_v_noise = self.n_adapt_2(k_v_noise)
        noise_fea_list.append(k_v_noise)
        k_v_kernel = self.k_adapt_3(k_v_kernel)
        kernel_fea_list.append(k_v_kernel)
        k_v_noise = self.n_adapt_3(k_v_noise)
        noise_fea_list.append(k_v_noise)
        k_v_kernel = self.k_adapt_4(k_v_kernel)
        kernel_fea_list.append(k_v_kernel)
        k_v_noise = self.n_adapt_4(k_v_noise)
        noise_fea_list.append(k_v_noise)
        k_v_kernel = self.k_adapt_5(k_v_kernel)
        kernel_fea_list.append(k_v_kernel)
        k_v_noise = self.n_adapt_5(k_v_noise)
        noise_fea_list.append(k_v_noise)
        #print(torch.equal(noise_fea_list[0],noise_fea_list[1]))                                
        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, kernel_fea_list[i], noise_fea_list[i]])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x#, k_v_kernel


#######################################################################################
#combine with CDSR SR network: degradation info and image content  
# don't change here=>another file: CDSR_arch.py
class CDSR(nn.Module):
    def __init__(self, args):
        super(CDSR, self).__init__()
        in_nc = 3
        out_nc = 3
        nf = 64
        gc = 32
        scale = int(args.scale[0])
        kernel_size = 21
        nb = 10
        embedding_length = 256

        self.scale = scale
        self.kernel_size = kernel_size

        self.compress = nn.Sequential(
			nn.Linear(embedding_length, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU()
		)

        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=256)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * (scale ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.PixelShuffle(scale))

    def forward(self, x, fea):
        # paddingBottom = int(np.ceil(h / self.ps) * self.ps - h)
        # paddingRight = int(np.ceil(w / self.ps) * self.ps - w)
        # x = torch.nn.functional.pad(x, [0, paddingRight, 0, paddingBottom], mode='reflect')

        fea_map = self.compress(fea)
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, fea_map])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)
        return out

#######################################################################################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out

class Encoder_2(nn.Module):
    def __init__(self):
        super(Encoder_2, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out
####################################################################
class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_c=3, nf=8, nb=3, stride=1, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        
        self.layer_1 = nn.Sequential(nn.Conv2d(in_c, nf, kernel_size=kw, stride=1, padding=padw), #3->8
            nn.LeakyReLU(0.2, True))

        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        nf_mult_prev = nf_mult
        #nf_mult = min(2 ** n, 8)

        self.layer_2 = nn.Sequential(nn.Conv2d(
                    nf * nf_mult_prev,
                    nf * 2,
                    kernel_size=kw,
                    stride=stride,
                    padding=padw,
                    bias=use_bias,),
                    norm_layer(nf * 2), 
                    nn.LeakyReLU(0.2, True)
                    ) # nf*1 -> nf*2

        self.layer_3 = nn.Sequential(nn.Conv2d(
                    nf * 2,
                    nf * 4,
                    kernel_size=kw,
                    stride=stride,
                    padding=padw,
                    bias=use_bias,),
                    norm_layer(nf * 4),
                     nn.LeakyReLU(0.2, True)
                     ) # nf*2 -> nf*4

        self.final_layer = nn.Conv2d(nf * 4, nf, kernel_size=kw, stride=1, padding=padw) # output 1 channel prediction map        

        '''kw = 3
        padw = 1
        sequence = [
            nn.Conv2d(in_c, nf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        for n in range(1, nb):  # gradually increase the number of filters nb=3
            nf_mult_prev = nf_mult #(1=1)
            nf_mult = min(2 ** n, 8) (n=1 nf_mult_prev=1)
            sequence += [
                nn.Conv2d(
                    nf * nf_mult_prev,
                    nf * nf_mult,
                    kernel_size=kw,
                    stride=stride,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(nf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** nb, 8)
        sequence += [
            nn.Conv2d(
                nf * nf_mult_prev,
                nf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(nf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(nf * nf_mult, nf, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)'''

    def forward(self, input):
        """Standard forward."""
        #return self.model(input)
        x_1 = self.layer_1(input)
        x_2 = self.layer_2(x_1)
        x_3 = self.layer_3(x_2)
        x_4 = self.final_layer(x_3)
        return x_1, x_2, x_3, x_4
####################################################################
class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = DASR(args)#.cuda()
        # self.patch_discriminator = PatchGANDiscriminator()
        # Encoder
        self.E_kernel = MoCo(base_encoder=Encoder)#.cuda()
        self.E_noise = MoCo(base_encoder=Encoder)
        
    def forward(self, x):
        if self.training:
            #x_img = x[0]
            #x_noise = x[1]
            #x_kernel = x[2]
            x_img = x_noise = x_kernel = x[0]
            # x_HR = x[3]
            x_query_kernel = x_img[:, 0, ...]                          # b, c, h, w
            x_key_kernel = x_kernel[:, 1, ...]                            # b, c, h, w
            x_query_noise = x_img[:, 0, ...]                          # b, c, h, w
            x_key_noise = x_noise[:, 1, ...]                            # b, c, h, w
            # degradation-aware represenetion learning
            fea_kernel, logits_kernel, labels = self.E_kernel(x_query_kernel, x_key_kernel)
            fea_noise, logits_noise, labels = self.E_noise(x_query_noise, x_key_noise)
            # degradation-aware SR
            sr, map_kernel = self.G(x_query_kernel, fea_kernel, fea_noise)
            # real = self.patch_discriminator(x_HR)
            #fake = self.patch_discriminator(sr)
            return sr, logits_kernel, logits_noise, labels, map_kernel
        else:
            x_img = x_noise = x_kernel = x[0]
            #fea_kernel = x[1]
            # degradation-aware represenetion learning
            fea_kernel = self.E_kernel(x_kernel, x_kernel)
            fea_noise = self.E_noise(x_noise, x_noise)
            # degradation-aware SR
            sr = self.G(x_img, fea_kernel, fea_noise)
            #sr, map_kernel = self.G(x_img, fea_kernel, fea_noise)

            return sr#, map_kernel
