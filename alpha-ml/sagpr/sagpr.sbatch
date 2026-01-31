#!/bin/bash -l
## run in /bin/bash NOTE the -l flag!
#
## do not join stdout and stderr
#SBATCH -o log.out
#SBATCH -e log.err
## name of the job
#SBATCH -J train
## execute job from the current working directory
## this is default slurm behavior
#SBATCH -D ./
## do not send mail
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --mem-per-cpu=1500
#SBATCH --partition=medium
#SBATCH -t 01:00:00

ulimit -s unlimited
NTIME=28000


module purge
module load intel/21.2.0 impi/2021.2  mkl/2021.2
module load anaconda/3/2021.05
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_HOME/compiler/2021.2.0/linux/compiler/lib/intel64_lin/:$MKL_HOME/lib/intel64/:$HOME/.local/lib

file='train.xyz'

python /u/lazerpo/TENSOAP/soapfast/scripts/split_dataset.py -f $file -n 25 -o coords


sagpr_get_PS -f $file -lm 0 -p -nc 1000 -ns 100 -sm 'random' -o PS0_details
for i in {0..24}
        do
        sagpr_get_PS -f coords_${i}.xyz -lm 0 -p -sf PS0_details -o PS0_${i} &
        done
wait
/u/lazerpo/TENSOAP/soapfast/scripts/rebuild_power_spectrum.py -lm 0 -c $file -nb 25 -f PS0


sagpr_get_PS -f $file -lm 2 -p -nc 1000 -ns 100 -sm 'random' -o PS2_details
wait
for i in {0..24}
        do
        sagpr_get_PS -f coords_${i}.xyz -lm 2 -p -sf PS2_details -o PS2_${i} &
        done
wait
/u/lazerpo/TENSOAP/soapfast/scripts/rebuild_power_spectrum.py -lm 2 -c $file -nb 25 -f PS2

/u/lazerpo/TENSOAP/soapfast/scripts/get_atomic_power_spectrum.py -p PS0.npy -f $file -o PS0_sparse_atomic
/u/lazerpo/TENSOAP/soapfast/scripts/get_atomic_power_spectrum.py -p PS2.npy -f $file -o PS2_sparse_atomic


if [[ -f "PS0_natoms.npy" && -f "PS2_natoms.npy"  ]]; then
        echo "Natoms files present"
        elif [ -f "natoms.npy" ]; then
                cp natoms.npy PS0_natoms.npy && cp natoms.npy PS2_natoms.npy
        else exit 0
fi

sagpr_do_env_fps -p PS0_sparse_atomic.npy -n 2000 -o FPS_0
sagpr_do_env_fps -p PS2_sparse_atomic.npy -n 2000 -o FPS_2
sagpr_apply_env_fps -p PS0_sparse_atomic.npy -sf FPS_0_rows -o PS0_full_sparsified_atomic
sagpr_apply_env_fps -p PS2_sparse_atomic.npy -sf FPS_2_rows -o PS2_full_sparsified_atomic
sagpr_get_kernel -ps PS0.npy PS0_full_sparsified_atomic.npy -s PS0_natoms.npy NONE -z 2 -o KERNEL_L0_NM
sagpr_get_kernel -ps PS0_full_sparsified_atomic.npy -s NONE -z 2 -o KERNEL_L0_MM

sagpr_get_kernel -ps PS2.npy PS2_full_sparsified_atomic.npy -ps0 PS0.npy PS0_full_sparsified_atomic.npy -s PS2_natoms.npy NONE -z 2 -o KERNEL_L2_NM
sagpr_get_kernel -ps PS2_full_sparsified_atomic.npy -ps0 PS0_full_sparsified_atomic.npy -s NONE -z 2 -o KERNEL_L2_MM

sagpr_train -r 2 -reg 1e-8 1e-5 -f $file -sf KERNEL_L0_NM.npy KERNEL_L0_MM.npy KERNEL_L2_NM.npy KERNEL_L2_MM.npy -p alpha -rdm 800 -m 'pinv' -t 1.0 -pr > out.txt 