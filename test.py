from option import args
import torch
import utility
import data
import model
import loss

#from trainer_modify_constraint import Trainer
#from trainer_modify_self_supervise_discriminator import Trainer
#from trainer_modify_motion_blur import Trainer
#from trainer_modify_two_branch import Trainer
#from trainer_CDSR import Trainer
#from trainer_modify_noise_as_kernel_extractor_input import Trainer
#from trainer import Trainer
from trainer_modify_constraint_regressor import Trainer
#from trainer_modify import Trainer
#from trainer_concat_with_fixed_representation import Trainer
#from trainer_modify_SRnetwork_cdsr import Trainer
#from trainer_ablation1 import Trainer
#from trainer_ablation2 import Trainer
#from trainer_modify_two_branch_fusion import Trainer
#from trainer_upper_bound import Trainer
#from trainer_denoise_dncnn import Trainer
#freom trainer_edsr import Trainer

import os


os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    if checkpoint.ok:
        loader = data.Data(args)
        #loader = data.motion_Data(args)
        model = model.Model(args, checkpoint)
        #print(model)
        #model = BlindSR(args).cuda()
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args,loader ,model, loss, checkpoint)
        while not t.terminate():
            t.test()
        checkpoint.done()
