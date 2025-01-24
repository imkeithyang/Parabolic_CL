# Parabolic Continual Learning
This is the code repository for our paper **Parabolic Continual Learning (AISTATS 2025)**. In this work, we consider a continual learning technique where the evolution of the learner's loss satisfies a parabolic partial differential equation (PDE) over the space of tasks. We enforce this through the stochastic representation of solutions to parabolic PDEs through a sampling technique. This provides a fast and easy-to-implement approach for learning new tasks. We then unify the expected loss with the task memory to qualitatively describe the error for new tasks. Our empirical findings suggest the method performs competitively on a variety of benchmarks. 

![figure](Figure1.pdf)

This repo is largely adapted from the [mammoth repository](https://github.com/aimagelab/mammoth/tree/master). Please follow the instructions on the mammoth repository to install necessary requirements. Our implementation of the parabolic continual learner is in ```models/er_parabolic.py```. Additionally, the algorithm to sample Brownian bridges is available in ```models/utils/brownian_utils.py```.

To run experiments, use the following command with the specified dataset and seed input:  
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

To run imbalanced dataset experiment or corrupted data experiment, add ```--imbalance 1``` or ```--label_shuffle 1``` to the command. 

