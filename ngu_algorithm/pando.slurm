#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --job-name=NGU
#MODULE LOAD
module load gcc
module load python/3.8
conda init bash
conda activate ray

NODES=($(scontrol show hostnames $SLURM_NODELIST))

#RUN SCRIPT
for i in {0..4}; do
# srun --exclusive -N1 -n1 -w ${NODES[0]} --output=Easy_${i}.txt \
# python3 run.py --env_class maze --env_name Easy --N 150 --gamma 0.99 --lr_theta 5e-4 \
# --lr_q 1e-3 --lr_starts 5e2 --tau 0.005 --policy_frequency 2 --ngu_frequency 5 --target_frequency 1 \
# --noise 0.5 --alpha 0.1 --autotune False --n_episode 8 --batch_size 256 --seed $i --device cpu \
# --n_env 5 --wandb_project_name LDS_1 --wandb_entity ngu_algorithm &
# srun --exclusive -N1 -n1 -w ${NODES[1]} --output=Ur_${i}.txt \
# python3 run.py --env_class maze --env_name Ur --N 150 --gamma 0.99 --lr_theta 5e-4 \
# --lr_q 1e-3 --lr_starts 5e2 --tau 0.005 --policy_frequency 2 --ngu_frequency 5 --target_frequency 1 \
# --noise 0.5 --alpha 0.1 --autotune False --n_episode 8 --batch_size 256 --seed $i --device cpu \
# --n_env 5 --wandb_project_name LDS_1 --wandb_entity ngu_algorithm &
# srun --exclusive -N1 -n1 -w ${NODES[2]} --output=Hard_${i}.txt \
# python3 run.py --env_class maze --env_name Hard --N 150 --gamma 0.99 --lr_theta 5e-4 \
# --lr_q 1e-3 --lr_starts 5e2 --tau 0.005 --policy_frequency 2 --ngu_frequency 5 --target_frequency 1 \
# --noise 0.5 --alpha 0.1 --autotune False --n_episode 8 --batch_size 256 --seed $i --device cpu \
# --n_env 5 --wandb_project_name LDS_1 --wandb_entity ngu_algorithm &

srun --exclusive -N1 -n1 -w ${NODES[0]} --output=fetch_${i}.txt \
python3 run.py --env_class fetch --env_name fetch --N 100 --gamma 0.99 --lr_theta 5e-4 \
--lr_q 1e-3 --lr_starts 5e2 --tau 0.005 --policy_frequency 2 --ngu_frequency 5 --target_frequency 1 \
--noise 0.5 --alpha 0.1 --autotune False --n_episode 8 --batch_size 256 --seed $i --device cpu \
--n_env 5 --wandb_project_name LDS_2 --wandb_entity ngu_algorithm &

srun --exclusive -N1 -n1 -w ${NODES[1]} --output=finger_${i}.txt \
python3 run.py --env_class finger --env_name finger --N 90 \
--gamma 0.99 --lr_theta 5e-4 --lr_q 1e-3 --lr_starts 5e2 --tau 0.005 --policy_frequency 2 \
--target_frequency 1 --ngu_frequency 5 --noise 0.5 --alpha 0.1 --autotune False --n_episode 8 --batch_size 256 \
--seed $i --device cpu --n_env 6 --wandb_project_name LDS_2 --wandb_entity ngu_algorithm &

# srun --exclusive -N1 -n1 -w ${NODES[1]} --output=hopper_${i}.txt \
# python3 run.py --env_class hopper --env_name hopper --N 50 \
# --gamma 0.99 --lr_theta 5e-4 --lr_q 1e-3 --lr_starts 5e2 --tau 0.005 --policy_frequency 2 \
# --target_frequency 1 --ngu_frequency 5 --noise 0.5 --alpha 0.1 --autotune False --n_episode 8 --batch_size 256 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_1 --wandb_entity ngu_algorithm &

srun --exclusive -N1 -n1 -w ${NODES[2]} --output=fetch_push_${i}.txt \
python3 run.py --env_class fetch_push --env_name fetch_push --N 84 \
--gamma 0.99 --lr_theta 5e-4 --lr_q 1e-3 --lr_starts 5e2 --tau 0.005 --policy_frequency 2 \
--target_frequency 1 --ngu_frequency 5 --noise 0.5 --alpha 0.1 --autotune False --n_episode 8 --batch_size 256 \
--seed $i --device cpu --n_env 6 --wandb_project_name LDS_2 --wandb_entity ngu_algorithm &

done
wait