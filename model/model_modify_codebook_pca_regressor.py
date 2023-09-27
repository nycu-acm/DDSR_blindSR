import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder_regressor import MoCo


def make_model(args):
    return BlindSR(args)

# seperate kernel and noise representation
# condition on kernel and noise representation
# concate noise and feature
# case1 DAB kernel as input first
# case2 DAB noise as input first
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


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
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

        # compress codebook1
        '''self.compress_kernel = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )
        self.compress_noise = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )'''
        #compress codebook2
        self.compress_kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )
        self.compress_noise = nn.Sequential(
            nn.Linear(64, 64, bias=False),
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

    def forward(self, x, k_v_kernel, k_v_noise):
        k_v_kernel = self.compress_kernel(k_v_kernel)
        k_v_noise = self.compress_noise(k_v_noise)
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
        '''self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        # codebook1
        '''self.feature_bank = nn.Embedding(1024, 256)
        self.fc_q = nn.Linear(256, 256)
        self.fc_k = nn.Linear(256, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        #codebook2
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )
        self.feature_bank = nn.Embedding(1024, 64)
        self.fc_q = nn.Linear(64, 64)
        self.fc_k = nn.Linear(64, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )        
    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea = self.mlp1(fea)
        q = self.fc_q(fea)
        fb = self.feature_bank.weight
        k = self.fc_k(fb)
        qk = torch.mm(q, k.transpose(1, 0))
        qk = qk/10
        qk = F.softmax(qk, dim=1)
        fea = torch.mm(qk, fb)
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
        '''self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        # codebook1
        '''self.feature_bank = nn.Embedding(1024, 256)
        self.fc_q = nn.Linear(256, 256)
        self.fc_k = nn.Linear(256, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        #codebook2
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )
        self.feature_bank = nn.Embedding(128, 64)
        self.fc_q = nn.Linear(64, 64)
        self.fc_k = nn.Linear(64, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )            
    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea = self.mlp1(fea)
        q = self.fc_q(fea)
        fb = self.feature_bank.weight
        k = self.fc_k(fb)
        q = nn.functional.normalize(q,dim=1)
        k = nn.functional.normalize(k,dim=1)
        qk = torch.mm(q, k.transpose(1, 0))
        #### temperature parameter = 10
        #qk = qk/10000 ##1/13 experiment pca04
        qk = F.softmax(qk, dim=1)
        #qk = F.relu(qk) ## experiment pca06
        #max_index = torch.argmax(qk)
        #print('max index:',max_index)
        #print('value:',qk[:,max_index])
        #value, idx = torch.sort(qk,descending=True)
        #print('first 10 index',idx[:,:10])        
        #print('first 10 value:',value[:,:10])
        #print(qk)
        fea = torch.mm(qk, fb)
        out = self.mlp(fea)
        return fea, out
class Encoder_2_regressor(nn.Module):
    def __init__(self):
        super(Encoder_2_regressor, self).__init__()

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
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        '''self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        # codebook1
        '''self.feature_bank = nn.Embedding(1024, 256)
        self.fc_q = nn.Linear(256, 256)
        self.fc_k = nn.Linear(256, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        #codebook2
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )
        self.feature_bank = nn.Embedding(128, 64)
        self.fc_q = nn.Linear(64, 64)
        self.fc_k = nn.Linear(64, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )            
    def forward(self, x):
        fea_map = self.E(x)
        fea = self.pool(fea_map).squeeze(-1).squeeze(-1)
        fea = self.mlp1(fea)
        q = self.fc_q(fea)        
        fb = self.feature_bank.weight
        k = self.fc_k(fb)
        q = nn.functional.normalize(q,dim=1)
        k = nn.functional.normalize(k,dim=1)        
        qk = torch.mm(q, k.transpose(1, 0))
        qk = F.softmax(qk, dim=1)
        qk = qk
        #max_index = torch.argmax(qk)
        #print(qk[:,max_index])
        #value, idx = torch.sort(qk,descending=True)
        #print(value[:,:5])
        #print(idx[:,:5])
        #print(qk)
        fea = torch.mm(qk, fb)
        out = self.mlp(fea)        

        return fea, out, fea_map
class Encoder_3_regressor(nn.Module):
    def __init__(self):
        super(Encoder_3_regressor, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(9+64, 64, kernel_size=3, padding=1),
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
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        '''self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        # codebook1
        '''self.feature_bank = nn.Embedding(1024, 256)
        self.fc_q = nn.Linear(256, 256)
        self.fc_k = nn.Linear(256, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )'''
        #codebook2
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )
        self.feature_bank = nn.Embedding(128, 64)
        self.fc_q = nn.Linear(64, 64)
        self.fc_k = nn.Linear(64, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )            
    def forward(self, x):
        fea_map = self.E(x).squeeze(-1).squeeze(-1)
        fea = self.pool(fea_map).squeeze(-1).squeeze(-1)
        fea = self.mlp1(fea)
        q = self.fc_q(fea)
        fb = self.feature_bank.weight
        k = self.fc_k(fb)
        q = nn.functional.normalize(q,dim=1)
        k = nn.functional.normalize(k,dim=1)
        qk = torch.mm(q, k.transpose(1, 0))
        #### temperature parameter = 10
        #qk = qk/10000 ##1/13 experiment pca04
        qk = F.softmax(qk, dim=1)
        #qk = F.relu(qk) ## experiment pca06
        #max_index = torch.argmax(qk)
        #print('max index:',max_index)
        #print('value:',qk[:,max_index])
        #value, idx = torch.sort(qk,descending=True)
        #print('first 10 index',idx[:,:10])        
        #print('first 10 value:',value[:,:10])
        #print(qk)
        fea = torch.mm(qk, fb)
        out = self.mlp(fea)
        return fea, out, fea_map
                
class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = DASR(args)#.cuda()

        # Encoder
        self.E_kernel = MoCo(base_encoder=Encoder_2_regressor)#.cuda()
        self.E_noise = MoCo(base_encoder=Encoder_2_regressor)
    def forward(self, x):
        if self.training:
            x_kernel = x[0]
            x_noise = x[1]
            x_query_kernel = x_kernel[:, 0, ...]                          # b, c, h, w
            x_key_kernel = x_kernel[:, 1, ...]                            # b, c, h, w
            x_query_noise = x_noise[:, 0, ...]                          # b, c, h, w
            x_key_noise = x_noise[:, 1, ...]                            # b, c, h, w
            # degradation-aware represenetion learning
            fea_kernel, logits_kernel, labels = self.E_kernel(x_query_kernel, x_key_kernel)
            fea_noise, logits_noise, labels = self.E_noise(x_query_noise, x_key_noise)
            # degradation-aware SR
            sr = self.G(x_query_kernel, fea_kernel, fea_noise)
            return sr, logits_kernel, logits_noise, labels
        else:
            x_kernel= x[1]
            x_noise = x[1]
            x_img = x[0]
            #fea_noise = x[2]
            #x_kernel = x_noise = x
            # degradation-aware represenetion learning
            fea_kernel = self.E_kernel(x_kernel, x_kernel)
            fea_noise = self.E_noise(x_noise, x_noise)
            # degradation-aware SR
            
            sr = self.G(x_img, fea_kernel, fea_noise)

            return sr
