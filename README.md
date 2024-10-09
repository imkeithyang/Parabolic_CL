# Parabolic Continual Learner
## Code base for Parabolic continual learner

The Parabolic PDE based continual learning algorithm is implemented in ```models/er_parabolic.py```. This code base is adapted from the continual learning benchmark repository [mammoth](https://github.com/aimagelab/mammoth); we directly benchmarked our method against the methods implemented in mammoth. Please follow the instructions on the [mammoth](https://github.com/aimagelab/mammoth) repository to install the necessary packages. 

Below is an example command to run experiments on Seq-CIFAR 10.      
```
python ./utils/main.py\
        --model er_parabolic\
        --dataset seq-cifar10\
        --n_epochs 1 (Online, class incremental scenario)\
        --batch_size 32\
        --minibatch_size 32\
        --buffer_mode reservoir\
        --lr 0.08\
        --sigma_x 0.03 (diffusion coefficient of data)\
        --sigma_y 0.01 (diffusion coefficient of labels)\
        --n_t 5 (number of time samples in Brownian bridges)\
        --n_b 1 (number of Brownian bridge samples)\
        --buffer_size (any size)\
        --device (cuda or cpu)\
        --seed (seed)\
        --label_shuffle (0 for no label corruption or 1 for 50% label corruption)
```
