import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder import MoCo


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

class denoising_DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(denoising_DAB, self).__init__()
        #noise as input paper method da_conv
        # noise noise as input concat noise and feature
        self.da_conv1 = conv(n_feat*2, n_feat, kernel_size)
        self.da_conv2 = conv(n_feat*2, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        self.relu =  nn.LeakyReLU(0.1, True)
    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation noise representation: B * C

        '''
        # noise noise as input concat noise and feature
        noise = torch.unsqueeze(x[1],-1)
        noise = torch.unsqueeze(noise,-1)
        noise = noise.repeat(1,1,x[0].shape[2],x[0].shape[3])
        out = torch.cat((x[0],noise),dim=1)
        out = self.relu(self.da_conv1(out))
        out = self.relu(self.conv1(out))
        out = torch.cat((x[0],noise),dim=1)
        out = self.relu(self.da_conv2(out))
        out = self.conv2(out) + x[0]
        return out


class denoising_DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(denoising_DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            denoising_DAB(conv, n_feat, kernel_size, reduction) \
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
        
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res




class denoising_net(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(denoising_net, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        # scale = int(args.scale[0])

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.noise_level_fc = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            denoising_DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(conv(n_feats,3,kernel_size))


    def forward(self, x, x_noise_level):
        x_noise_level = self.noise_level_fc(x_noise_level)
        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, x_noise_level])
        res = self.body[-1](res)
        res = res + x
        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)
        return x

class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
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
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class deblur_DASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(deblur_DASR, self).__init__()

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
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

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

    def forward(self, x, k_v):
        k_v = self.compress(k_v)

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x

##########################################
class modify_DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(modify_DA_conv, self).__init__()
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


class modify_CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(modify_CA_layer, self).__init__()
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


# additionanl class conditional instance normalization
class CIN(nn.Module):
  def __init__(self,latent_size=32):
    super(CIN,self).__init__()
    self.scalar =  nn.Sequential(nn.Linear(64, latent_size, bias=False),nn.LeakyReLU(0.1, True),nn.Linear(latent_size, 64, bias=False)) 
    self.bias = nn.Sequential(nn.Linear(64, latent_size, bias=False),nn.LeakyReLU(0.1, True),nn.Linear(latent_size, 64, bias=False))
  def calc_mean_std(self,feat, eps=1e-5):
      # eps is a small value added to the variance to avoid divide-by-zero.
      size = feat.size()
      #print(size)
      assert (len(size) == 4)
      N, C = size[:2]
      feat_var = feat.view(N, C, -1).var(dim=2) + eps
      feat_std = feat_var.sqrt().view(N, C, 1, 1)
      feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
      return feat_mean, feat_std
  def forward(self, x):
    feat, noise = x[0], x[1]
    feat_mean, feat_std = self.calc_mean_std(feat)
    size = feat.size()
    noise_scalar = self.scalar(noise).view(size[0],size[1],1,1)
    noise_bias =self.bias(noise).view(size[0],size[1],1,1)
    affine_feat = noise_scalar.expand(size)*(feat-feat_mean.expand(size))/feat_std.expand(size) + noise_bias.expand(size)
    #affine_feat = (feat-feat_mean.expand(size))/feat_std.expand(size)
    return affine_feat


class modify_DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(modify_DAB, self).__init__()

        self.da_conv1 = modify_DA_conv(n_feat, n_feat, kernel_size, reduction)
        #self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = conv(n_feat*2, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        self.relu =  nn.LeakyReLU(0.1, True)
        self.cin = CIN(32)
    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation kernel representation: B * C
        :param x[2]: degradation noise representation: B * C
        '''
        #print(x[0].shape)
        #print(x[1].shape)
        #print(x[2].shape)
        # kernel as input first concat noise and feature
        '''x_in = [x[0], x[1]]
        out = self.relu(self.da_conv1(x_in))
        out = self.relu(self.conv1(out))
        noise = torch.unsqueeze(x[2],-1)
        noise = torch.unsqueeze(noise,-1)
        noise = noise.repeat(1,1,out.shape[2],out.shape[3])
        out = torch.cat((out,noise),dim=1)
        out = self.relu(self.da_conv2(out))
        #out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]
        return out'''
        #noise as input first concat noise and feature
        noise = torch.unsqueeze(x[2],-1)
        noise = torch.unsqueeze(noise,-1)
        noise = noise.repeat(1,1,x[0].shape[2],x[0].shape[3])
        out = torch.cat((x[0],noise),dim=1)
        out = self.relu(self.da_conv2(out))
        out = self.relu(self.conv1(out))
        x_in = [out,x[1]]
        out = self.relu(self.da_conv1(x_in))                        
        #out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]
        return out
        '''#nonise as input first conditional instance normalization
        x_in = [x[0],x[2]]
        out = self.relu(self.cin.forward(x_in))
        out = self.relu(self.conv1(out))
        x_in2 = [out,x[1]]
        out = self.relu(self.da_conv1(x_in2))                        
        #out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]
        return out'''       


class modify_DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(modify_DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            modify_DAB(conv, n_feat, kernel_size, reduction) \
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
        
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1], x[2]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class modify_DASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(modify_DASR, self).__init__()

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
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            modify_DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v_kernel, k_v_noise):
        k_v_kernel = self.compress(k_v_kernel)
        k_v_noise = self.compress(k_v_noise)
        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v_kernel, k_v_noise])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x
###########################################
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


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        #self.model_deblur = deblur_DASR(args)
        self.model_deblur = modify_DASR(args)
        self.model_denoise = denoising_net(args)
        # Encoder
        #self.E = MoCo(base_encoder=Encoder)
        self.E_kernel = MoCo(base_encoder=Encoder)#.cuda()
        self.E_noise = MoCo(base_encoder=Encoder)
    def forward(self, x):
        if self.training:
            x_query = x[:, 0, ...]                          # b, c, h, w
            x_key = x[:, 1, ...]                            # b, c, h, w

            # degradation-aware represenetion learning
            #fea_noise, logits_noise, labels_noise = self.E_noise(x_query, x_key)
            fea_noise = self.E_noise(x_query, x_key)
            lr_denoise_1= self.model_denoise(x_query,fea_noise)            
            lr_denoise_2 = self.model_denoise(x_key,fea_noise)
            #fea_kernel, logits_kernel, labels_kernel = self.E_kernel(lr_denoise_1, lr_denoise_2)
            fea_kernel = self.E_kernel(lr_denoise_1, lr_denoise_2)
            sr = self.model_deblur(lr_denoise_1,fea_kernel,fea_noise)
            return sr, lr_denoise_1
        else:
        
            fea_noise = self.E_noise(x, x)
            lr_denoise_1= self.model_denoise(x,fea_noise)
            fea_kernel = self.E_kernel(lr_denoise_1, lr_denoise_1)
            sr = self.model_deblur(lr_denoise_1,fea_kernel)
            return sr,lr_denoise_1
