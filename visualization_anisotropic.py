import torch
from option import args
#from model.blindsr_modify_concat_with_fixed_representation import BlindSR
#from model.blindsr import BlindSR
#from model.model_modify_codebook_pca import BlindSR
#from model.model_modify_codebook_pca_regressor import BlindSR
#from model.CDSR import BlindSR
from model.model_ablation1 import BlindSR
#from model.model_modify_SRnetwork_cdsr import BlindSR
#from model.model_two_branch_fusion import BlindSR
#from model.model_two_branch import BlindSR
import glob
import imageio
import numpy as np
from utils import util
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.pyplot import cm
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from pytorch_wavelets import DWTForward, DWTInverse
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = 'cuda:0'
# degradation settings
args.scale = [4]
args.blur_kernel = 21
args.blur_type = 'aniso_gaussian'
#args.downsampler = 's-fold'
#args.blur_type = 'iso_gaussian'
lambda_1_list = [0.5, 4.0, 2.0, 3.2]
lambda_2_list = [0.5, 4.0, 1.0, 1.5]
theta_list    = [0,   0,   30,  135]
noise_list = [0,5,10,15]
#noise_list = [0,5,10,15]
noise_type_list =['gaussian','gaussian','poisson','poisson']#,'poisson & gaussian']
sigma_list =[0.2, 1.0, 1.8, 2.6]
blur_size_list= [5.0,11.0,15.0,21.0]
angle_list = [-70,-50,0,60]
noise_list2 = [30,10,10,20]
# paths
img_path = '/mnt/HDD3/yuan/dataset/test/B100/*.png'
#img_path = '/mnt/HDD3/yuan/dataset/test/DIV2K_test_LR_unknown/X4/*.png'
xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()  # Accepts all wave types available to PyWavelets
label = []
for i in range(len(sigma_list)):
      #label.append(str(sigma_list[i]))
      label.append('['+str(lambda_1_list[i])+','+str(lambda_2_list[i])+','+str(theta_list[i])+']')
print(label)

if __name__ == '__main__':
    net = BlindSR(args).to(device)
    #net.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10/model_580.pt'), strict=False)
    #net.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10_noise_as_kernel_extractor_input02/model_600.pt'), strict=False)
    
    #net.load_state_dict(torch.load('experiment/blindsr_x4_bicubic_aniso/model/pretrain_model_600.pt'), strict=False)
    #net.load_state_dict(torch.load('/mnt/HDD8/yuan/model_CDSR01/model_501.pt'), strict=False)
    net.load_state_dict(torch.load('/mnt/HDD8/yuan/model_ablation1/model_300.pt'), strict=False)
    #net.load_state_dict(torch.load("/mnt/HDD3/yuan/CDSR/experiment/cdsrn15_x4_bicubic_aniso/model/model_414.pt"),strict=False)
    #net.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10_cdsr_range/model_550.pt'), strict=False)
    #net.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10_twobranch03/model_650.pt'), strict=False)
    #net.load_state_dict(torch.load('/mnt/HDD8/yuan/model_DASR_cdsr_range/model_550.pt'), strict=False)
    '''pretrained_dict = torch.load('/mnt/HDD8/yuan/model_concat_with_fixed_representation03/model_300.pt')
    model_dict = net.state_dict()
    for k,v in pretrained_dict.items():
      if 'G' not in k:
        print(k)         
        pretrained_dict = {k:v}
        #pretrained_dict = {k:v  for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)'''
    net.eval()

    HR_img_list = sorted(glob.glob(img_path))
    
    fea_list = []
    fea_list_kernel = []
    avg_kernel = 0
    
    for lambda_1, lambda_2, theta,noise in zip(lambda_1_list, lambda_2_list, theta_list,noise_list):
    #for lambda_1, lambda_2, theta,noise, noise_type  in zip(lambda_1_list, lambda_2_list, theta_list,noise_list,noise_type_list):
    #for noise in noise_list:
    #for sigma in sigma_list:
        '''degrade = util.SRMDPreprocessing(
            scale=args.scale[0],
            kernel_size=args.blur_kernel,
            blur_type=args.blur_type,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            #lambda_1=np.random.uniform(0,4),
            #lambda_2=np.random.uniform(0,4),
            #sig = sigma,
            theta=theta,
            noise=0,#noise,
            #noise_type=noise_type
        )'''
        print('==================================================================')
        with torch.no_grad():
            for i in range(len(HR_img_list)):
                degrade = util.SRMDPreprocessing(
                      scale=args.scale[0],
                      kernel_size=args.blur_kernel,
                      blur_type=args.blur_type,
                      lambda_1=lambda_1,
                      lambda_2=lambda_2,
                      #lambda_1=np.random.uniform(0,4),
                      #lambda_2=np.random.uniform(0,4),
                      #sig = sigma,
                      theta =theta,#np.random.uniform(0,180),
                      noise=0,#noise,#np.random.uniform(0,15)
                      #noise_type=noise_type
                    )         
                # read HR images
                #print(HR_img_list[i])
                HR_img = imageio.imread(HR_img_list[i])
                #HR_img = imageio.imread('/mnt/HDD3/yuan/dataset/BSDS100/108070.png')
                if np.ndim(HR_img) < 3:
                    HR_img = np.stack([HR_img, HR_img, HR_img], 2)
                HR_img = np.ascontiguousarray(HR_img.transpose((2, 0, 1)))
                HR_img = torch.from_numpy(HR_img).float().to(device).unsqueeze(0).unsqueeze(1)
                b, n, c, h, w = HR_img.size()
                HR_img = HR_img[:, :, :, :h // args.scale[0] * args.scale[0], :w // args.scale[0] * args.scale[0]]
                
                # generate LR images
                LR_img, _,_ = degrade(HR_img, random=False)
                LR_img = LR_img.to(device)
                Yl, Yh = xfm(LR_img[:,0,...])
                #print(Yl.shape)
                #for i in range(len(Yh)):
                  #print(Yh[i].shape)
                Yh1= Yh[0][:,0,...]
                Yh2= Yh[0][:,1,...]
                Yh3= Yh[0][:,2,...]
                #LR_img =torch.cat((Yl,Yh1,Yh2,Yh3),1)
                LR_img_freq =torch.cat((Yh1,Yh2,Yh3),1)
                #LR_img = Yh3
                #fea_noise = net.E_noise(LR_img[:,0,...], LR_img[:,0,...])
                #lr_denoise_1= net.model_denoise(LR_img[:,0,...],fea_noise)
                #_,fea = net.E_kernel.encoder_q(lr_denoise_1)
                #fea_2 = net.E_kernel(lr_denoise_1,lr_denoise_1)
                #lr_denoise_1 = torch.squeeze(torch.clamp(lr_denoise_1,0,255)).permute(1,2,0)
                #lr_denoise_1 = lr_denoise_1.detach().cpu().numpy()
                #cv2.imwrite('/mnt/HDD3/yuan/DASR/dn_lr_'+str(i)+'_.png',cv2.cvtColor(lr_denoise_1, cv2.COLOR_RGB2BGR))
                #print(LR_img)
                # generate degradation representations
                #_, fea_2 = net.E.encoder_q(LR_img[:, 0, ...])
                #_, fea_2 = net.E.encoder_q(LR_img[:, 0, ...]/255.)
                #_,fea_2 = net.E_kernel.encoder_q(LR_img[:, 0, ...])
                #fea_n = net.E_noise(LR_img_freq,LR_img_freq)
                #noise_k = torch.unsqueeze(fea_n,-1)
                #noise_k = torch.unsqueeze(noise_k,-1)
                #noise_k = noise_k.repeat(1,1,LR_img_freq.shape[2],LR_img_freq.shape[3])
                #lr_k_k_cat = torch.cat((LR_img_freq,noise_k),dim=1)                  
                #fea_2 = net.E_kernel(lr_k_k_cat,lr_k_k_cat)
                #fea_2 = net.E_kernel(LR_img_freq,LR_img_freq)
                #print(LR_img_freq.shape)
                
                #sr = net.G(LR_img[:,0,...],fea_2,fea_2)
                fea_2 = net.E_kernel(LR_img[:, 0, ...],LR_img[:, 0, ...])
                #fea_2 = net.E(LR_img[:, 0, ...]/255,LR_img[:, 0, ...]/255)
                #fea_2 = net.E_noise(LR_img[:, 0, ...],LR_img[:, 0, ...])
                #fea_2 = net.E_noise(LR_img_freq,LR_img_freq)
                # cdsr
                #b, n, c, h,w = LR_img.shape
                #LR_img = LR_img.view(b,c,h,w)
                #fea_2 = net.E(LR_img,LR_img)
                #print(fea_2.shape)
                #print(fea.shape)
                #print(fea_2)
                fea_list.append(fea_2.data.cpu().numpy())
                #avg_kernel+=fea_2#kernel_fea
                #fea_list_kernel.append(fea.data.cpu().numpy())
                #print('pair',i,'distance',distance.euclidean(fea_1.data.cpu().numpy(), fea_2.data.cpu().numpy()))
                #print('Cosine similarity:',cosine_similarity(fea_1.data.cpu().numpy(), fea_2.data.cpu().numpy()).item())
            #print(len(fea_list))
            #avg_kernel=avg_kernel/len(HR_img_list)
            #fea_list_kernel.append(avg_kernel.data.cpu().numpy())
    """for blur_size,angle,noise,noise_type in zip(blur_size_list, angle_list, noise_list2,noise_type_list):
    #for lambda_1, lambda_2, theta,noise, noise_type  in zip(lambda_1_list, lambda_2_list, theta_list,noise_list,noise_type_list):
    #for noise in noise_list:
    #for sigma in sigma_list:
        '''degrade = util.SRMDPreprocessing(
            scale=args.scale[0],
            kernel_size=args.blur_kernel,
            blur_type=args.blur_type,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            #lambda_1=np.random.uniform(0,4),
            #lambda_2=np.random.uniform(0,4),
            #sig = sigma,
            theta=theta,
            noise=0,#noise,
            #noise_type=noise_type
        )'''
        print('==================================================================')
        with torch.no_grad():
            for i in range(len(HR_img_list)):
                degrade = util.motion_SRMDPreprocessing(
                      scale=args.scale[0],
                      kernel_size=11,#args.blur_kernel,
                      blur_type=args.blur_type,
                      #lambda_1=0,#lambda_1,
                      #lambda_2=0,#lambda_2,
                      lambda_1=np.random.uniform(0,4),
                      lambda_2=np.random.uniform(0,4),
                      #sig = sigma,
                      theta =0,#theta,#np.random.uniform(0,180),
                      noise=noise,#np.random.uniform(0,25),
                      noise_type=noise_type,
                      #angle = angle,
                      #blur_size=blur_size
                      angle = 0,#np.random.randint(-90,90),
                      blur_size=10,#np.random.randint(2,10),
                      
                    )         
                # read HR images
                #print(HR_img_list[i])
                HR_img = imageio.imread(HR_img_list[i])
                #HR_img = imageio.imread('/mnt/HDD3/yuan/dataset/BSDS100/108070.png')
                if np.ndim(HR_img) < 3:
                    HR_img = np.stack([HR_img, HR_img, HR_img], 2)
                HR_img = np.ascontiguousarray(HR_img.transpose((2, 0, 1)))
                HR_img = torch.from_numpy(HR_img).float().to(device).unsqueeze(0).unsqueeze(1)
                b, n, c, h, w = HR_img.size()
                HR_img = HR_img[:, :, :, :h // args.scale[0] * args.scale[0], :w // args.scale[0] * args.scale[0]]
                
                # generate LR images
                LR_img, _ = degrade(HR_img, random=False)
                LR_img = LR_img.to(device)
                Yl, Yh = xfm(LR_img[:,0,...])
                #print(Yl.shape)
                #for i in range(len(Yh)):
                  #print(Yh[i].shape)
                Yh1= Yh[0][:,0,...]
                Yh2= Yh[0][:,1,...]
                Yh3= Yh[0][:,2,...]
                #LR_img =torch.cat((Yl,Yh1,Yh2,Yh3),1)
                LR_img_freq =torch.cat((Yh1,Yh2,Yh3),1)
                #LR_img = Yh3
                #fea_noise = net.E_noise(LR_img[:,0,...], LR_img[:,0,...])
                #lr_denoise_1= net.model_denoise(LR_img[:,0,...],fea_noise)
                #_,fea = net.E_kernel.encoder_q(lr_denoise_1)
                #fea_2 = net.E_kernel(lr_denoise_1,lr_denoise_1)


                # generate degradation representations
                #_, kernel_fea = net.E.encoder_q(LR_img[:, 0, ...])
                #_,fea_2 = net.E_kernel.encoder_q(LR_img[:, 0, ...])
                #fea_2 = net.E_kernel(LR_img_freq,LR_img_freq)
                #fea_2 = net.E_kernel(LR_img[:, 0, ...],LR_img[:, 0, ...])
                #fea_2 = net.E(LR_img[:, 0, ...],LR_img[:, 0, ...])
                #fea_2 = net.E_noise(LR_img[:, 0, ...],LR_img[:, 0, ...])
                fea_2 = net.E_noise(LR_img_freq,LR_img_freq)
                fea_list.append(fea_2.data.cpu().numpy())
          
    '''for i in range(len(fea_list_kernel)-1):
        for j in range(i+1,len(fea_list_kernel)):
            print('[{},{}]\t'
                  'Euclidean distance:{:.6f}'.format(
                  i,j,
                  distance.euclidean(fea_list_kernel[i], fea_list_kernel[j]))
                  )
            print('Cosine similarity:',cosine_similarity(fea_list_kernel[i], fea_list_kernel[j]).item())
    #print(a)'''      
   #print(a)"""
    '''for i in range(4):
        with torch.no_grad():   
          if i ==0:
            img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/DIV2K_train_HR/lr_unknown_val/0801x4.png')
            #img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/NTIRE2020_Track1/Corrupted-va-x/0801.png')            
          if i ==1:
            img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/DIV2K_train_HR/lr_unknown_val/0811x4.png')
            #img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/NTIRE2020_Track1/Corrupted-va-x/0855.png')
          if i ==2:
            img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/DIV2K_train_HR/lr_unknown_val/0866x4.png')
            #img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/NTIRE2020_Track1/Corrupted-va-x/0889.png')                      
          else:
            img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/DIV2K_train_HR/lr_unknown_val/0888x4.png')
            #img_blur = cv2.imread('/mnt/HDD3/yuan/dataset/NTIRE2020_Track1/Corrupted-va-x/0896.png')
          print(i)
          #print(img_blur.shape)
          img_blur = np.transpose(img_blur,(2,0,1))        
          c, h, w = img_blur.shape
          patch_h = round(h/10)
          patch_w = round(w/10)
          #print(patch_h,patch_w)
                 
          # begin (x,y) left_top
          range_x = np.arange(0, w, step=patch_w)
          range_y = np.arange(0, h, step=patch_h)
          #print(range_y.shape)
          #print(range_x.shape) 
          TMP = 0 
          for y in range_y:
              for x in range_x:
                  patch_blur = img_blur[:,y:y + patch_h, x:x + patch_w]      
                  c, h, w = patch_blur.shape
                  patch_blur = np.expand_dims(patch_blur,axis=0)
                  patch_blur = np.expand_dims(patch_blur,axis=0)
                  B,N,C, H, W = patch_blur.shape
                  patch_blur = torch.from_numpy(patch_blur).float().to(device)
                  #noise_level_2 = torch.rand(1, 1, 1, 1, 1).to(patch_blur.device) * 25
                  #noise_2 = torch.randn_like(patch_blur).view(-1, N, C, H, W).mul_(noise_level_2).view(-1, C, H, W)                  
                  #patch_blur.add_(noise_2)                  
                  Yl, Yh = xfm(patch_blur[:,0,...])
                  Yh1= Yh[0][:,0,...]
                  Yh2= Yh[0][:,1,...]
                  Yh3= Yh[0][:,2,...]
                  LR_img_freq =torch.cat((Yh1,Yh2,Yh3),1)
                  fea_2 = net.E_kernel(LR_img_freq,LR_img_freq)
                  #fea_2 = net.E_noise(LR_img_freq,LR_img_freq)
                  if TMP<100:
                    fea_list.append(fea_2.data.cpu().numpy())
                  TMP+=1                  
                  #print(TMP)'''     
    f = np.concatenate(fea_list,0) 
    f_min = np.min(f, 0)
    f_max = np.max(f, 0)
    # normalization
    f_norm = (f - f_min) / (f_max - f_min)
    # T-SNE
    tsne = TSNE(n_components=2,learning_rate=200, init='pca', random_state=0)
    #pca = PCA(n_components=2)
    embed = tsne.fit_transform(f)
    #embed = pca.fit_transform(f)
    embed = embed.reshape(len(lambda_1_list), 1, 100, -1)
    #embed = embed.reshape(len(sigma_list), 1, len(HR_img_list), -1)
    # visualization
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    #ax = plt.subplot(111,projection='3d')
    #ax = plt.subplot(121)
#embed(#of kernel,1,#of image,dim of feature)
    color=cm.rainbow(np.linspace(0,1,len(HR_img_list)))
    label = []
    #for i in range(len(go_denoise_list)):
    for i in range(len(sigma_list)):
      #label.append(str(sigma_list[i]))
      label.append('['+str(lambda_1_list[i])+','+str(lambda_2_list[i])+','+str(theta_list[i])+']')
      #label.append(str(noise_list[i]))
      #label.append('img_'+str(i))
    '''for i in range(len(sigma_list)):
      #label.append(str(sigma_list[i]))
      #label.append(str(lambda_1_list[i])+str(lambda_2_list[i]))
      #label.append(noise_type_list[i]+str(noise_list[i]))
      label.append('img_'+str(i))'''
    #for i in range(len(f)-1):
    #  for j in range(1,len(f)):
    #    print((f[i]==f[j]).all())
      
    '''for i in range(len(HR_img_list)):
    #for i in range(3):
       
       a1=plt.scatter(embed[0, 0, i, 0], embed[0, 0, i, 1],c=color[i].reshape(1,-1),marker='o',alpha=0.35, label=label[0])#, label=str(i)+'img')
       a2=plt.scatter(embed[1, 0, i, 0], embed[1, 0, i, 1],c=color[i].reshape(1,-1),marker='x',alpha=0.35, label=label[1])#, label=str(i)+'img')
       a3=plt.scatter(embed[2, 0, i, 0], embed[2, 0, i, 1],c=color[i].reshape(1,-1),marker='*',alpha=0.35, label=label[2])#, label=str(i)+'img')
       a4=plt.scatter(embed[3, 0, i, 0], embed[3, 0, i, 1],c=color[i].reshape(1,-1),marker='s',alpha=0.35, label=label[3])#, label=str(i)+'img')
       #a5=plt.scatter(embed[4, 0, i, 0], embed[4, 0, i, 1],c=color[i].reshape(1,-1),marker='^',alpha=0.35, label=label[4])#, label=str(i)+'img')
       plt.text(embed[0, 0, i, 0], embed[0, 0, i, 1],str(i))
       plt.text(embed[1, 0, i, 0], embed[1, 0, i, 1],str(i))
       plt.text(embed[2, 0, i, 0], embed[2, 0, i, 1],str(i))
       plt.text(embed[3, 0, i, 0], embed[3, 0, i, 1],str(i))
       #plt.text(embed[4, 0, i, 0], embed[4, 0, i, 1],str(i))
       #ax.plot(embed[0, 0, i, 0], embed[0, 0, i, 1], embed[0, 0, i, 2],c=color[i],marker='o', label=str(i)+'img')
       #ax.plot(embed[1, 0, i, 0], embed[1, 0, i, 1], embed[1, 0, i, 2],c=color[i],marker='x', label=str(i)+'img')
       #ax.plot(embed[2, 0, i, 0], embed[2, 0, i, 1], embed[2, 0, i, 2],c=color[i],marker='.', label=str(i)+'img')
       #ax.plot(embed[3, 0, i, 0], embed[3, 0, i, 1], embed[3, 0, i, 2],c=color[i],marker='s', label=str(i)+'img')
    #plt.legend([a1,a2,a3,a4,a5],[str(sigma_list[0]),str(sigma_list[1]),str(sigma_list[2]),str(sigma_list[3]),str(sigma_list[4])])
    plt.legend([a1,a2,a3,a4],label)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)'''
    ax = plt.subplot(111)
    a1=plt.scatter(embed[0, 0, :, 0], embed[0, 0, :, 1], c='b')
    a2=plt.scatter(embed[1, 0, :, 0], embed[1, 0, :, 1], c='r')
    a3=plt.scatter(embed[2, 0, :, 0], embed[2, 0, :, 1], c='g')
    a4=plt.scatter(embed[3, 0, :, 0], embed[3, 0, :, 1], c='k')
    #a5=plt.scatter(embed[4, 0, :, 0], embed[4, 0, :, 1], c='purple')
    #a6=plt.scatter(embed[5, 0, :, 0], embed[5, 0, :, 1], c='cyan')
    #a7=plt.scatter(embed[6, 0, :, 0], embed[6, 0, :, 1], c='navy')
    #a8=plt.scatter(embed[7, 0, :, 0], embed[7, 0, :, 1], c='salmon')    
    #a5=plt.scatter(embed[4, 0, :, 0], embed[4, 0, :, 1], c='purple')
    #a6=plt.scatter(embed[5, 0, :, 0], embed[5, 0, :, 1], c='cyan')    
    #plt.legend([a1,a2,a3,a4,a5,a6,a7,a8],label)
    #plt.legend([a1,a2,a3,a4],label)
    #plt.savefig('CDSR_dk_wo_n.png',dpi=100)    
    plt.show()
    
