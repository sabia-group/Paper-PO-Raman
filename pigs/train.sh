#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=p.ada
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=none

#SBATCH --time=24:00:00
#SBATCH -J trainAnthra
#SBATCH -o job.out
#SBATCH -e job.err

module purge
module load anaconda/3/2021.11 cuda/11.6
conda activate mace_alpha
module load cudnn/8.8.1 pytorch/gpu-cuda-11.6/2.0.0 gcc/11 openmpi/4 mpi4py/3.0.3
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 -u /u/lazerpo/mace_alpha/scripts/run_train.py \
    --name="TePIGS_model" \
    --train_file="../dataset.xyz" \
    --valid_file="../dataset.xyz" \
    --test_file="../dataset.xyz" \
    --E0s='average' \
    --model="MACE" \
    --hidden_irreps='64x0e' \
    --r_max=3.0 \
    --batch_size=1 \
    --max_num_epochs=1000 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cpu \
    --forces_key='delta_force' \
    --energy_weight=0.0 \
    --scaling='no_scaling'
