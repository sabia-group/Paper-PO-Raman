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
    --name="MACE_mu_alpha" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --model="AtomicDipolesMACE" \
    --num_channels=32 \
    --max_L=2 \
    --r_max=5.0 \
    --loss="dipole" \
    --dipole_weight=0.0 \
    --polarizability_weight=100.0 \
    --dipole_key="dipole" \
    --polarizability_key="alpha" \
    --batch_size=16 \
    --valid_batch_size=32 \
    --max_num_epochs=2000 \
    --scheduler_patience=6 \
    --patience=12 \
    --ema \
    --error_table='DipoleRMSE' \
    --default_dtype="float64"\
    --device=cuda \
    --seed=123 \
    --restart_latest \
    --save_cpu \

sleep 5
