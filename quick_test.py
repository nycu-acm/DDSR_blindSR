from model.blindsr import BlindSR
#from model.model_modify_codebook_pca_regressor import BlindSR
#from model.model_modify_codebook_pca import BlindSR
#from model.CDSR import BlindSR
#from model.edsr import EDSR
#from model.model_ablation1 import BlindSR
#from model.model_modify_SRnetwork_cdsr import BlindSR
import torch
import numpy as np
import imageio
import argparse
import os
import utility
import cv2
import glob
from tqdm import tqdm
import csv
#from skimage import color
#import lpips
from pytorch_wavelets import DWTForward, DWTInverse

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
'''def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/test/',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='2',
                        help='super resolution scale')
    parser.add_argument('--resume', type=int, default=600,
                        help='resume from specific checkpoint')
    parser.add_argument('--blur_type', type=str, default='aniso_gaussian',
                        help='blur types (iso_gaussian | aniso_gaussian)')
    return parser.parse_args()'''


def main(image_path,args):
    #args = parse_args()
    if args.blur_type == 'iso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_iso'
    elif args.blur_type == 'aniso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso'
    else:
      dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso'
    dataset = image_path.split('/')[-2]
    scale = args.scale
    xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()   # Accepts all wave types available to PyWavelets
    ifm = DWTInverse( mode='zero', wave='haar').cuda() 
    #print(a) 
    # path to save sr images
    #save_dir = dir + '/results/'+dataset
    #save_dir = '/mnt/HDD8/yuan/results_freq_kernel07_contrastive_codebook_pca10/'+dataset+'_w_n10'
    #save_dir = '/mnt/HDD8/yuan//results_dasr_pretrain/'+dataset#+'_w_n10'
    #save_dir = '/mnt/HDD8/yuan/EDSR/results_edsr/'+dataset +'_w_n10' 
    #save_dir = '/mnt/HDD8/yuan/EDSR/results_edsr/'+dataset 
    save_dir = '/mnt/HDD8/yuan/results_lr/'+dataset+'_w_n10'  
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    #loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    #DASR = EDSR(args).cuda() #BlindSR(args).cuda()
    DASR = BlindSR(args).cuda()
    DASR.load_state_dict(torch.load('/mnt/HDD3/yuan/DASR/experiment/blindsr_x4_bicubic_aniso/model/pretrain_model_600.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10_SRnetwork_cdsr03/model_' + str(args.resume) + '.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10_cdsr_range/model_550.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10_fine_tune_on_17Track2_02/model_630.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_DASR_cdsr_range/model_550.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/EDSR/ptrtrain/EDSR_x4.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_ablation1/model_350.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD3/yuan/CDSR/experiment/cdsrn15_x4_bicubic_aniso/model/model_' + str(args.resume) + '.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10/model_' + str(args.resume) + '.pt'), strict=False)
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_freq_kernel07_contrastive_codebook_pca10_random_downsample01/model_' + str(args.resume) + '.pt'), strict=False) 
      
    #DASR.load_state_dict(torch.load('/mnt/HDD8/yuan/model_CDSR01/model_' + str(args.resume) + '.pt'), strict=False)
    DASR.eval()
    lr = imageio.imread(image_path)
    
    #print(lr.shape)
    if lr.ndim==2:
      lr = np.stack((lr, lr, lr), axis=-1)
    lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
    lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)

    b,N,C,H,W =lr.shape
    noise_level=10
    noise = torch.randn_like(lr).view(-1, N, C, H, W).mul_(noise_level)
    lr = lr.add_(noise)
    #lr = lr/255.
    Yl, Yh = xfm(lr[:,0,...]) # positive sample with noise
    Yh1= Yh[0][:,0,...] #HL
    Yh2= Yh[0][:,1,...] #LH
    Yh3= Yh[0][:,2,...] #HH
    lr_wavelet =torch.cat((Yh1,Yh2,Yh3),1)    
    # inference
    #sr = DASR([lr[:, 0,:-1, ...],lr_wavelet])
    #sr = DASR([lr[:, 0, ...],lr_wavelet])
    #sr = DASR([lr[:, 0, ...],lr[:, 0, ...]])
    filename =image_path.split('/')[-1]
    '''if filename=='Canon_046_LR4.png' or filename=='Canon_047_LR4.png':
      print(filename)
      DASR.to('cpu')
      sr = DASR(lr[:, 0, ...].to('cpu'))
      sr = sr.cuda()
      DASR.cuda()
    else:
      with torch.no_grad():
        sr = DASR(lr[:, 0, ...])'''
    sr = DASR(lr[:, 0,:-1, ...])
    #sr = DASR(lr[:, 0, ...])
    #print(lr.shape)
    #DIV2KRK
    #sr = DASR(lr[:, 0,:-1, ...])
    #sr = sr*255
    #sr = DASR(lr[:, 0, ...])
    sr = utility.quantize(sr, 255.0)
    filename =image_path.split('/')[-1]
    # DIV2KRK
    filename = filename.replace('.png','_gt.png')
    filename = filename.replace('im','img')
    # NTIRE2017 Track2    
    #filename = filename.replace('x4.','.') 
    #filename = filename.replace('_x4.0_SR.','.') 
    # RealSRv3
    #filename = filename.replace('LR4','HR')   
    if args.evaluate:
      hr = imageio.imread(args.hr_path+filename)
      hr = np.ascontiguousarray(hr.transpose((2, 0, 1)))
      hr = torch.from_numpy(hr).float().cuda().unsqueeze(0) #.unsqueeze(0)
      hr = hr[:, :-1,...]
      #print(sr.shape)
      #print(hr.shape)
      #d_lpips = loss_fn_alex(sr, hr)
      psnr_ = utility.calc_psnr(
                        sr, hr, int(scale), 255,
                        benchmark=False
                    )
      ssim_ = utility.calc_ssim(
                        sr, hr, int(scale),
                        benchmark=False
                    )
      #print(args.hr_path+image_path.split('/')[-1])
      
      #print('PSNR:{:.3f} SSIM:{:.4f} LPIPS:{:.4f}'.format(psnr_,ssim_,d_lpips.item()))
    # save sr results
    img_name = image_path.split('.png')[0].split('/')[-1]
    sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
    sr = sr[:, :, [2, 1, 0]]
    #print(lr[:,0, :-1,...].shape)
    #lr = np.array(lr[:,0, :-1,...].squeeze(0).permute(1, 2, 0).data.cpu())
    #print(save_dir)
    cv2.imwrite(save_dir + '/' + img_name + '_sr.png', Sr)
    return psnr_, ssim_#, d_lpips.item()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/NTIRE2020_Track1/Corrupted-va-x/', help='image directory')
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/test/historical/LR/', help='image directory')
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/AIM19Track2/valid-input-noisy/', help='image directory')
  parser.add_argument('--filePath', type=str, default='/mnt/HDD3/wingho/datasets/DIV2KRK/lr_x4/', help='image directory')
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/RealSR_V3/Canon/Canon/4/LRx4_Canon/', help='image directory')
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/RealSR_V3/Nikon/Nikon/4/LRx4_Nikon/', help='image directory')
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD8/yuan/results_lr/Set14_ablation/', help='image directory')    
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/NTIRE2018_Track2/DIV2K_valid_LR_mild/', help='image directory')
  #parser.add_argument('--filePath', type=str, default='/mnt/HDD3/yuan/dataset/DIV2K_train_HR/lr_unknown_val/', help='image directory')
  parser.add_argument('--scale', type=str, default='4', help='super resolution scale') 
  parser.add_argument('--resume', type=int, default=585, help='resume from specific checkpoint')
  parser.add_argument('--blur_type', type=str, default='aniso_gaussian', help='blur types (iso_gaussian | aniso_gaussian)')
  parser.add_argument('--evaluate',default=True)#False)
  #parser.add_argument('--hr_path', type=str, default='/mnt/HDD3/yuan/dataset/DIV2K_train_HR/val/')
  parser.add_argument('--hr_path', type=str, default='/mnt/HDD3/wingho/datasets/DIV2KRK/gt/')
  #parser.add_argument('--hr_path', type=str, default='/mnt/HDD8/yuan/results_hr/Set14_ablation/')
  #parser.add_argument('--hr_path', type=str, default='/mnt/HDD3/yuan/dataset/RealSR_V3/Canon/Canon/4/HR/')
  #parser.add_argument('--hr_path', type=str, default='/mnt/HDD3/yuan/dataset/RealSR_V3/Nikon/Nikon/4/HR/')
  
  parser.add_argument('--model', default='EDSR',help='model name')
  parser.add_argument('--act', type=str, default='relu',help='activation function')
  parser.add_argument('--pre_train', type=str, default='.',help='pre-trained model directory')
  parser.add_argument('--extend', type=str, default='.',help='pre-trained model directory')
  parser.add_argument('--n_resblocks', type=int, default=16,help='number of residual blocks')
  parser.add_argument('--n_feats', type=int, default=64,help='number of feature maps')
  parser.add_argument('--res_scale', type=float, default=1,help='residual scaling')
  parser.add_argument('--shift_mean', default=True,help='subtract pixel mean from the input')
  parser.add_argument('--dilation', action='store_true',help='use dilated convolution')
  parser.add_argument('--precision', type=str, default='single',choices=('single', 'half'),help='FP precision for test (single | half)')  
  parser.add_argument('--rgb_range', type=int, default=255,help='maximum value of RGB')  
  parser.add_argument('--n_colors', type=int, default=3,help='number of color channels to use')  
  
  args = parser.parse_args()
  with torch.no_grad():
    '''ignore_list = ['X3','X2']
    file_list = os.listdir(args.filePath)
    for item in ignore_list:
      file_list.remove(item)
    for file_name in tqdm(file_list):
      test_list = glob.glob(args.filePath+file_name+"/*")'''
    test_list = sorted(glob.glob(args.filePath+"/*.png"))
    eval_psnr = 0
    eval_ssim = 0
    eval_lpips = 0
    psnr_ = 0
    ssim_ = 0   
    str_ = args.blur_type.split("_")[0]
    csv_path = './experiment/blindsr_x' + str(int(args.scale[0]))+'_bicubic_'+ str_+'/results_pdm/ori_evaluate.csv'
      
    for image in test_list:
      psnr_,ssim_ =main(image,args)
      #main(image,args)
      eval_psnr += psnr_
      eval_ssim += ssim_
      #eval_lpips += lpips_
      filename=image.split('/')[-1]
      #print(filename,'PSNR:{:.3f} SSIM{:.4f} LPIPS{:.4f}'.format(psnr_,ssim_,lpips_))
      print(filename,'PSNR:{:.3f} SSIM{:.4f}'.format(psnr_,ssim_))
    print(eval_psnr/len(test_list),eval_ssim/len(test_list))#,eval_lpips/len(test_list))