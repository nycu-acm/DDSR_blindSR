from option import args
import torch
import utility
import data
import model
import loss

from trainer_modify_constraint_regressor import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    #print(args)
    if checkpoint.ok:
        #loader = data.motion_Data(args)
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        #print(kernel_opt)
        #model = BlindSR(args).cuda()
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        #t = Trainer(args, loader, model, loss, checkpoint,kernel_opt, noise_opt)
        t = Trainer(args, loader, model ,loss, checkpoint)
        while not t.terminate():
            t.train()
        checkpoint.done()
