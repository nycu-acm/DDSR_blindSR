import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
import csv
import lpips
import cv2
from tensorboardX import SummaryWriter
from pytorch_wavelets import DWTForward, DWTInverse
import random 
#os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

# extract kernel and noise seperately
# non end-to-end train
# first train the degradation model 300 epochs
# then train the SR network 300 epoch
# kernel encoder: postive sample (same kernel) negative sample (differnet kernel)
# output model_new1 server:114 (concat noise representation and feature) kernel as input first modify /model//model_modify.py class DAB
# output model_new2 server: 148 (concat noise representation and feature) noise as input first  modify /model//model_modify.py class DAB
# output model_new3 server 148 (conditional instance normalization) noise as input first  modify /model//model_modify.py class DAB
# output model_new4 server 148 noise include gaussian.poisson.jpeg noise (concat noise representation and feature) kernel as input first modify /model//model_modify.py class DAB
# output model_new5 server 148 noise include gaussian.poisson (concat noise representation and feature) kernel as input first modify /model//model_modify.py class DAB 
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        #self.model_G = torch.nn.DataParallel(self.model.get_model().G, range(self.args.n_GPUs))
        #self.model_E_kernel = torch.nn.DataParallel(self.model.get_model().E_kernel, range(self.args.n_GPUs))
        #self.model_E_noise = torch.nn.DataParallel(self.model.get_model().E_noise, range(self.args.n_GPUs))
        self.model_G = self.model.get_model().G
        self.model_E_kernel = self.model.get_model().E_kernel
        self.model_E_noise = self.model.get_model().E_noise   
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        tensorboard_folder = './tensorboard_ablation3' #'./tensorboard_freq_kernel07_contrastive_codebook_pca10_fine_tune_on_17Track2/'
        self.tb_logger_1 = SummaryWriter(log_dir= os.path.join(tensorboard_folder , "loss_constrastive"))
        self.tb_logger_2 = SummaryWriter(log_dir= os.path.join(tensorboard_folder , "loss_sr"))
        self.loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
        self.xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()  # Accepts all wave types available to PyWavelets
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()
        if args.resume>0:
          #self.iteration = args.resume*len(self.loader_train.dataset)/self.args.batch_size
          self.iteration = args.resume*round(31050/self.args.batch_size+0.5)
        else:
          self.iteration = 0
    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder #* (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        '''degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )'''
        degrade = util.my_SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )
        timer = utility.timer()
        losses_contrast_kernel, losses_contrast_noise, losses_sr,losses_regression_kernel, losses_regression_noise = utility.AverageMeter(), utility.AverageMeter(), utility.AverageMeter(), utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            hr = hr.cuda()                              # b, n, c, h, w
            #lr, b_kernels = degrade(hr)                 # bn, c, h, w
            hr_0 = hr[:,0,...]
            hr_1 = hr[:,1,...]
            hr_1 = torch.cat((hr_1[2:,...],hr_1[:2,...]), dim=0)
            hr = torch.stack((hr_0,hr_1),dim=1)
            #print(hr.shape)
            lr_blured_kernel, lr_no_blured, lr_blured_kernel_noise_1,lr_no_blured_noise,lr_blured_kernel_noise_2, b_kernel = degrade(hr)                 # b, n, c, h, w
            #print(lr_blured_kernel[0,0,...].cpu().detach().numpy().shape)
            #cv2.imwrite('hr.png',hr[0,0,...].detach().cpu().numpy().transpose(1,2,0))
            #cv2.imwrite('lr.png',lr_blured_kernel[0,0,...].cpu().detach().numpy().transpose(1,2,0))
            #print(a)
            Yl, Yh = self.xfm(lr_blured_kernel_noise_1[:,0,...]) # positive sample with noise
            Yh1= Yh[0][:,0,...] #HL
            Yh2= Yh[0][:,1,...] #LH
            Yh3= Yh[0][:,2,...] #HH
            lr_blured_kernel_noise_1_01 =torch.cat((Yh1,Yh2,Yh3),1)
            lr_blured_kernel_noise_1_01 = lr_blured_kernel_noise_1_01.detach()
            Yl, Yh = self.xfm(lr_blured_kernel[:,1,...]) # positive sample with noise
            Yh1= Yh[0][:,0,...]
            Yh2= Yh[0][:,1,...]
            Yh3= Yh[0][:,2,...]
            lr_blured_kernel_noise_2_02 = torch.cat((Yh1,Yh2,Yh3),1)
            lr_blured_kernel_noise_2_02 = lr_blured_kernel_noise_2_02.detach()      
            if batch %2==0:
              tmp = lr_blured_kernel_noise_1_01
              lr_blured_kernel_noise_1_01 = lr_blured_kernel_noise_2_02
              lr_blured_kernel_noise_2_02 = tmp            
            Yl, Yh = self.xfm(lr_blured_kernel_noise_2[:,0,...])
            Yh1= Yh[0][:,0,...]
            Yh2= Yh[0][:,1,...]
            Yh3= Yh[0][:,2,...]
            lr_blured_kernel_noise_2_01 = torch.cat((Yh1,Yh2,Yh3),1)
            lr_blured_kernel_noise_2_01 = lr_blured_kernel_noise_2_01.detach()            
            Yl, Yh = self.xfm(lr_no_blured_noise[:,1,...])
            Yh1= Yh[0][:,0,...]
            Yh2= Yh[0][:,1,...]
            Yh3= Yh[0][:,2,...]
            lr_no_blured_noise_2_02 = torch.cat((Yh1,Yh2,Yh3),1)
            lr_no_blured_noise_2_02 = lr_no_blured_noise_2_02.detach()
            lr_q = lr_blured_kernel_noise_2_01
            lr_k = lr_no_blured_noise_2_02            
            self.optimizer.zero_grad()
            if batch %2==0:
              tmp = lr_q
              lr_q = lr_k
              lr_k = tmp                               
            timer.tic()
            # forward
            ## train degradation encoder
            if epoch <= self.args.epochs_encoder:
                _, output_kernel, target_kernel = self.model_E_kernel(im_q=lr_blured_kernel_noise_1_01, im_k=lr_blured_kernel_noise_2_02)
                _, output_noise, target_noise = self.model_E_noise(im_q=lr_q, im_k=lr_k)
                # embedding(degradation representation of query), logits(negative sample predict), labels(for negative sample: 0 )
                loss_constrast_kernel = self.contrast_loss(output_kernel, target_kernel)
                loss_constrast_noise = self.contrast_loss(output_noise, target_noise)
                loss = loss_constrast_kernel + loss_constrast_noise
                losses_contrast_kernel.update(loss_constrast_kernel.item())
                losses_contrast_noise.update(loss_constrast_noise.item())               
                if self.iteration %100 ==0:
                  self.tb_logger_1.add_scalar( "L_kernel" , loss_constrast_kernel.item() , self.iteration)
                  self.tb_logger_1.add_scalar( "L_noise" , loss_constrast_noise.item() , self.iteration)
                ## train the whole network
            elif epoch <= -1:#self.args.epochs_encoder+375:
                self.model_E_kernel.eval()
                self.model_E_noise.eval()
                with torch.no_grad():
                  embeding_1 = self.model_E_kernel(im_q=lr_blured_kernel_noise_2_01, im_k=lr_blured_kernel_noise_2_02)
                  embeding_2 = self.model_E_noise(im_q=lr_blured_kernel_noise_2_01, im_k=lr_k)
                #print(embeding_1.shape)                
                sr = self.model_G(lr_blured_kernel_noise_2[:,0,...], embeding_1, embeding_2)
                Yl, Yh = self.xfm(hr[:,0,...])
                Yh1= Yh[0][:,0,...]
                Yh2= Yh[0][:,1,...]
                Yh3= Yh[0][:,2,...]
                hr_wavelet = torch.cat((Yh1,Yh2,Yh3),1)
                Yl, Yh = self.xfm(sr)
                Yh1= Yh[0][:,0,...]
                Yh2= Yh[0][:,1,...]
                Yh3= Yh[0][:,2,...]
                sr_wavelet = torch.cat((Yh1,Yh2,Yh3),1)                            
                #self.model_E_kernel.eval()
                #self.model_E_noise.eval()
                #hr_wavelet = torch.cat((hr_wavelet[2:,...],hr_wavelet[:2,...]), dim=0)
                with torch.no_grad():              
                  _, sr_degradation_kernel, sr_fea_map_kernel = self.model_E_kernel.encoder_q(sr_wavelet)
                  _, hr_degradation_kernel, hr_fea_map_kernel = self.model_E_kernel.encoder_q(hr_wavelet)
                  #_, sr_degradation_noise, sr_fea_map_noise = self.model_E_noise.encoder_q(sr_wavelet)
                  #_, hr_degradation_noise, hr_fea_map_noise = self.model_E_noise.encoder_q(hr_wavelet)                  
                #loss_regression = self.loss(sr_degradation, hr_degradation) # pca09
                #loss_regression = self.loss(sr_fea_map, hr_fea_map)  # pca10
                ## pca11
                loss_regression_kernel = self.loss(sr_fea_map_kernel, hr_fea_map_kernel)
                #loss_regression_noise = self.loss(sr_fea_map_noise, hr_fea_map_noise)
                ## pca12              
                #loss_regression_kernel = self.loss(sr_degradation_kernel, hr_degradation_kernel)
                #loss_regression_noise = self.loss(sr_degradation_noise, hr_degradation_noise)                
                #print(loss_regression.item())              
                loss_SR = self.loss(sr, hr[:,0,...])
                print(loss_SR.item())
                loss = loss_SR + 10000*loss_regression_kernel #+ 10000*(loss_regression_kernel+loss_regression_noise)/2                
                #self.tb_logger_1.add_scalar( "L_kernel" , loss_constrast_kernel.item() , iteration)
                #self.tb_logger_1.add_scalar( "L_noise" , loss_constrast_noise.item() , iteration)
                if self.iteration %100 == 0:
                  self.tb_logger_2.add_scalar( "L_SR" , loss_SR.item() , self.iteration)
                  self.tb_logger_1.add_scalar( "L_regression_kernel" , 10000*loss_regression_kernel.item(), self.iteration)
                  #self.tb_logger_1.add_scalar( "L_regression_noise" , 10000*loss_regression_noise.item(), self.iteration)
                losses_sr.update(loss_SR.item())
                losses_regression_kernel.update(10000*loss_regression_kernel.item())
                #losses_regression_noise.update(10000*loss_regression_noise.item())               
            else:
              #for param in self.model_E_kernel.encoder_q.parameters():
                #param.requires_grad = False
              #  print(param.requires_grad)
              embeding_1, output_kernel, target_kernel = self.model_E_kernel(im_q=lr_blured_kernel_noise_1_01, im_k=lr_blured_kernel_noise_2_02)
              embeding_2, output_noise, target_noise = self.model_E_noise(im_q=lr_q, im_k=lr_k)
              sr = self.model_G(lr_blured_kernel_noise_2[:,0,...], embeding_1, embeding_2)
              Yl, Yh = self.xfm(hr[:,0,...])
              Yh1= Yh[0][:,0,...]
              Yh2= Yh[0][:,1,...]
              Yh3= Yh[0][:,2,...]
              hr_wavelet = torch.cat((Yh1,Yh2,Yh3),1)
              Yl, Yh = self.xfm(sr)
              Yh1= Yh[0][:,0,...]
              Yh2= Yh[0][:,1,...]
              Yh3= Yh[0][:,2,...]
              sr_wavelet = torch.cat((Yh1,Yh2,Yh3),1)
              hr_wavelet = torch.cat((hr_wavelet[2:,...],hr_wavelet[:2,...]), dim=0)                            
              #self.model_E_kernel.eval()
              #self.model_E_noise.eval()
              with torch.no_grad():              
                  sr_degradation_kernel, sr_mlp_degradation_kernel, sr_fea_map_kernel = self.model_E_kernel.encoder_q(sr_wavelet)
                  hr_degradation_kernel, hr_mlp_degradation_kernel, hr_fea_map_kernel = self.model_E_kernel.encoder_q(hr_wavelet)
                  #_, sr_degradation_noise, sr_fea_map_noise = self.model_E_noise.encoder_q(sr_wavelet)
                  #_, hr_degradation_noise, hr_fea_map_noise = self.model_E_noise.encoder_q(hr_wavelet)  
              #loss_regression = self.loss(sr_degradation, hr_degradation) # pca09
              #loss_regression = self.loss(sr_fea_map, hr_fea_map)  # pca10
              ## pca11
              loss_regression_kernel = self.loss(sr_fea_map_kernel, hr_fea_map_kernel)  
              #loss_regression_noise = self.loss(sr_fea_map_noise, hr_fea_map_noise)
              ## pca12              
              #loss_regression_kernel = self.loss(sr_degradation_kernel, hr_degradation_kernel)
              #loss_regression_noise = self.loss(sr_degradation_noise, hr_degradation_noise)                 
              #sr, output_kernel, output_noise, target = self.model([lr_same_kernel, lr_different_kernel])
              loss_SR = self.loss(sr, hr[:,0,...])
              #print(loss_SR.item())
              loss_constrast_kernel = self.contrast_loss(output_kernel, target_kernel)
              loss_constrast_noise = self.contrast_loss(output_noise, target_noise)
              loss = loss_SR + loss_constrast_kernel + loss_constrast_noise + 10000*loss_regression_kernel #+ 10000*(loss_regression_kernel+loss_regression_noise)/2 
              if self.iteration %100 == 0:
                self.tb_logger_2.add_scalar( "L_SR" , loss_SR.item() , self.iteration)
                self.tb_logger_1.add_scalar( "L_regression_kernel" , 10000*loss_regression_kernel.item(), self.iteration)
                #self.tb_logger_1.add_scalar( "L_regression_noise" , 10000*loss_regression_noise.item(), self.iteration)
                self.tb_logger_1.add_scalar( "L_kernel" , loss_constrast_kernel.item() , self.iteration)
                self.tb_logger_1.add_scalar( "L_noise" , loss_constrast_noise.item() , self.iteration)
              losses_sr.update(loss_SR.item())
              losses_contrast_kernel.update(loss_constrast_kernel.item())
              losses_contrast_noise.update(loss_constrast_noise.item())
              losses_regression_kernel.update(10000*loss_regression_kernel.item())
              #losses_regression_noise.update(10000*loss_regression_noise.item()) 
            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()
            self.iteration += 1
            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [contrastive loss kernel: {:.3f}]|contrastive loss noise: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_contrast_kernel.avg,
                            losses_contrast_noise.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [contrastive loss kernel: {:.3f}]|contrastive loss noise: {:.3f}]\t'
                        'Loss [SR loss:{:.3f}]\t'
                        'Loss [|Regression loss kernel:{:.6f}|Regression loss noise:{:.6f}]'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_contrast_kernel.avg,
                            losses_contrast_noise.avg,
                            losses_sr.avg,
                            losses_regression_kernel.avg,losses_regression_noise.avg,
                            timer.release()
                        ))
        self.loss.end_log(len(self.loader_train))
        # save model
        target = self.model.get_model()
        model_dict = target.state_dict()
        keys = list(model_dict.keys())
        #for key in keys:
            #if 'E.encoder_k' in key or 'queue' in key:
            #if 'E_noise' in key or 'G' in key:
            #    del model_dict[key]
        torch.save(
            model_dict,
            #os.path.join(self.ckp.dir, 'model_freq_kernel07_contrastive_codebook_pca03', 'model_{}.pt'.format(epoch))
            #os.path.join('/mnt/HDD8/yuan/', 'model_freq_kernel07_contrastive_codebook_pca10_fine_tune_on_17Track2', 'model_{}.pt'.format(epoch))
            os.path.join('/mnt/HDD8/yuan/', 'model_ablation3', 'model_{}.pt'.format(epoch))
        )

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        str_ = self.args.blur_type.split("_")[0]
        # ====================================================

          
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0
                eval_lpips = 0
                degrade = util.SRMDPreprocessing(
                                      self.scale[0],
                                      kernel_size=self.args.blur_kernel,
                                      blur_type=self.args.blur_type,
                                      sig=self.args.sig,
                                      lambda_1=self.args.lambda_1,
                                      lambda_2=self.args.lambda_2,
                                      theta=self.args.theta,
                                      noise=self.args.noise
                                    )         
                noise_ = torch.rand(1,255)
                
                for idx_img, (hr, filename, _) in enumerate(self.loader_test):
                    lambda_1 = random.choice([1.,5.,9.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
                    lambda_2 = random.choice([1.,5.,9.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
                    theta = random.choice([0.,45.])
                    #print(lambda_1,lambda_2,theta)
                    '''degrade = util.SRMDPreprocessing(
                      self.scale[0],
                      kernel_size=self.args.blur_kernel,
                      blur_type=self.args.blur_type,
                      sig=self.args.sig,
                      lambda_1=1,#lambda_1,#self.args.lambda_1,
                      lambda_2=1,#lambda_2,#self.args.lambda_2,
                      theta=theta,#self.args.theta,
                      noise=self.args.noise
                    )'''                   
                    hr = hr.cuda()                      # b, 1, c, h, w
                    hr = hr[:,:,[2,1,0],...]
                    hr = self.crop_border(hr, scale)
                    #lr_blured_noise10_1,lr_blured_noise10_2,lr_blured_noise20_1,lr_blured_noise20_2 = degrade(hr, random=False)   # b, 1, c, h, w
                    lr, kernel,_ = degrade(hr, random=False)   # b, 1, c, h, w 
                    hr = hr[:, 0, ...]                  # b, c, h, w
                    # inference
                    timer_test.tic()
                    #lr = lr_blured_noise10_1
                    Yl, Yh = self.xfm(lr[:,0,...])
                    #Yl, Yh = self.xfm(lr_blured_noise10_2[:,0,...])
                    Yh1= Yh[0][:,0,...] #HL
                    Yh2= Yh[0][:,1,...] #LH
                    Yh3= Yh[0][:,2,...] #HH
                    lr_freq =torch.cat((Yh1,Yh2,Yh3),1)                                        
                    sr= self.model([lr[:,0,...], lr_freq])
                    #print(idx_img)
                    '''if idx_img ==0:
                      lr_input = lr[:,0,...]
                      hr_compare = hr
                    fea_kernel = self.model_E_kernel(lr_freq_kernel, lr_freq_kernel)
                    fea_noise = self.model_E_noise(lr_freq_noise, lr_freq_noise)
                    sr = self.model_G(lr_input, fea_kernel, fea_noise)'''                    
                    timer_test.hold()
                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)
                    d_lpips = self.loss_fn_alex(sr, hr)
                    eval_lpips += d_lpips.item()
                    # metrics
                    psnr_ = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_psnr += psnr_ 
                    ssim_ = utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += ssim_

                    print(filename,'PSNR:{:.3f} SSIM{:.4f} LPIPS{:.4f}'.format(psnr_,ssim_,d_lpips.item()))                  
                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale,self.args.data_test,self.args.lambda_1,self.args.lambda_2,self.args.theta,self.args.noise)
                        #print(filename)
                self.ckp.save_results('kernel', [kernel], scale,self.args.data_test,self.args.lambda_1,self.args.lambda_2,self.args.theta,self.args.noise,False)
                print('kernel[',self.args.lambda_1,',',self.args.lambda_2,',',self.args.theta,']','noise level:',self.args.noise)
                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} LPIPS:{:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                        eval_lpips/ len(self.loader_test)
                    ))

    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()
        img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]
        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr