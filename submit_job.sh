# sbatch --account=bfhg-delta-gpu --partition=gpuA100x4 --mem=240g --tasks=1 --nodes=1 --gpus-per-node=4 --tasks-per-node=1 --cpus-per-task=16 --time=0-20:00:00 ./run_scripts/03_train_stage1.sh
# Stage 2: all SBATCH options are embedded in slurm_stage2.sh; it self-resubmits on walltime.
sbatch run_scripts/slurm_stage2.sh
# sbatch --account=bfhg-delta-gpu --partition=gpuA100x4 --mem=240g --tasks=1 --nodes=1 --gpus-per-node=4 --tasks-per-node=1 --cpus-per-task=16 --time=2-00:00:00  ./run_scripts/slurm_stage2.sh

# sbatch --account=bfhg-delta-gpu --partition=gpuA40x4 --mem=240g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=1 --cpus-per-task=16 --time=0-08:00:00 ./run_scripts/05_eval_vsibench.sh

# sbatch --account=bfhg-delta-gpu --partition=gpuA100x4 --mem=240g --tasks=1 --nodes=1 --gpus-per-node=2 --tasks-per-node=1 --cpus-per-task=16 --time=00:10:00 ./run_scripts/03_train_stage1.sh



# srun -A bfhg-dt-gh --partition=gpuA100x4 --gres=gpu:V100 --time=00:10:00 --tasks=4 --nodes=1 --ntasks-per-node=16 --pty ./run_scripts/teacher_finetune.sh
# srun --account=bfhg-delta-gpu --partition=gpuA100x4-interactive --mem=128g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=1 --cpus-per-task=8 --time=01:00:00 --pty bash
# srun --account=bfhg-delta-gpu --partition=gpuA40x4-interactive --mem=128g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=1 --cpus-per-task=8 --time=01:00:00 --pty bash
# srun --account=bfhg-delta-gpu --partition=cpu --mem=128g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=5 --cpus-per-task=16 --time=00:20:00 --pty bash

