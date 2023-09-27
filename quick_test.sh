# super-resolve an LR image (x2) using the model trained on noise-free degradations with isotropic Gaussian blurs
python quick_test.py --img_dir='/mnt/HDD3/yuan/dataset/Set14/LRbicx4/baboon.png' \
                     --scale='2' \
                     --resume=600 \
                     --blur_type='iso_gaussian'

# super-resolve an LR image (x4) using the model trained on general degradations with anisotropic Gaussian blurs and noises
python quick_test.py --img_dir='/mnt/HDD3/yuan/dataset/Set14/LRbicx4/baboon.png' \
                     --scale='4' \
                     --resume=600 \
                     --blur_type='aniso_gaussian'

cmd /k
