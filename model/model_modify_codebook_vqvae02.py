import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder_codebook_vqvae import MoCo


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
####################################################################################
class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta 
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
    
    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())
    
    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        # reshape z -> (batch, height, width, channel) and flatten
        #z = z.permute(0, 2, 3, 1).contiguous()
        #z_flattened = z.view(-1, self.e_dim)
        z_flattened = z.contiguous()
        codebook = self.embedding.weight
        d = self.dist(z_flattened, codebook)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean() 
            texture_loss = self.gram_loss(z, z_q_gt.detach()) 
            codebook_loss = codebook_loss + texture_loss 
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = z_q.contiguous()
        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1)
    
    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)        
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q
####################################################################################
###codebook (VQVAE v1)
'''class DASR(nn.Module):
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
        self.compress_noise = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )
        self.quantize_kernel = VectorQuantizer(
                n_e=1024,
                e_dim=64,
        )
        self.quantize_noise = VectorQuantizer(
                n_e=256,
                e_dim=64,
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
        #quantize codebook
        kernel_quant, kernel_codebook_loss, kernel_indices = self.quantize_kernel(k_v_kernel)
        noise_quant, noise_codebook_loss, noise_indices = self.quantize_noise(k_v_noise)
        
        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, kernel_quant, noise_quant])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x, kernel_codebook_loss, noise_codebook_loss, kernel_indices, noise_indices, kernel_quant, noise_quant'''
#codebook(VQVAE v2)
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
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )        
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
        )       
        # codebook
        self.quantize = VectorQuantizer(
                n_e=1024,
                e_dim=64,
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea = self.mlp1(fea)
        fea_quant, codebook_loss, indices = self.quantize(fea)      
        out = self.mlp(fea_quant)
        return fea_quant, out,codebook_loss, indices

class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = DASR(args)#.cuda()

        # Encoder
        self.E_kernel = MoCo(base_encoder=Encoder_2)#.cuda()
        self.E_noise = MoCo(base_encoder=Encoder_2)
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
            sr, kernel_codebook_loss, noise_codebook_loss, kernel_indices, noise_indices, kernel_quant, noise_quant = self.G(x_img, fea_kernel, fea_noise)

            return sr, kernel_codebook_loss, noise_codebook_loss, kernel_indices, noise_indices, kernel_quant, noise_quant
