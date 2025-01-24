# Parabolic continual learner
This is the github repository for our paper **Parabolic Continual Learner** which is accepted to AISTATS 2025. The parabolic continual learner constrained the loss of a continual learner to a parabolic PDE. We then used the Feynman-Kac theorem and applies Brownian Bridges to develop a scalable optimization.

This repo is largely adapted from the [mammoth repository](https://github.com/aimagelab/mammoth/tree/master). Please follow the instruction on the mammoth repository to install necessary requirements. Our implementation of the parabolic continual learner is in ```models/er_parablic.py```. Additionally, the algorithm to sample Brownian Bridges is provided in ```models/utils/brownian_utils.py```.

To run experiments use the following command,     
```
python ./utils/main.py \
        --model er_parabolic \
        --dataset $dataset \
        --n_epochs 1 \
        --batch_size 32 \
        --minibatch_size 32 \
        --buffer_size 1000 \
        --buffer_mode reservoir \
        --lr 0.08 \
        --sigma_x 0.03 \
        --sigma_y 0.01 \
        --n_t 5 \
        --n_b 1 \
        --weight 0 \
        --seed $seed
```

To run imbalance dataset experiment or corrupted data experiment, add ```--imbalance 1``` or ```--label_shuffle 1``` to the command. 

