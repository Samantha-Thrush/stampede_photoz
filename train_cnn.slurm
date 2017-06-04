#!/bin/bash
#SBATCH -J train_cnn_sam
#SBATCH -o train_cnn.%j.out
#SBATCH -e train_cnn.%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thrush2@illinois.edu

module load python
module load cuda

export THEANO_FLAGS='cuda.root="$TACC_CUDA_DIR",device=gpu,floatX=float32,nvcc.fastmath=True'

export PATH=$HOME:$WORK:$HOME/Lasagne/lasagne:$PATH

cd /work/04323/thrush/photoz

ibrun python train_cnn.py 
