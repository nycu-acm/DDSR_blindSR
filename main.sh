# general degradations with anisotropic Gaussian blurs and noises
python main.py --dir_data='/mnt/HDD3/yuan/dataset' \
               --scale='4' \
               --blur_type='aniso_gaussian' \
               --noise=25.0 \
               --epochs_encoder=200 \
               --epochs_sr=400\
               --n_GPUs=1 \
               --model='model_modify_codebook_pca_regressor' \
               --lambda_min=0.2 \
               --lambda_max=4.0 

#              -resume=575 \
#              --start_epoch=575 \
