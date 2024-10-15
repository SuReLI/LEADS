#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --job-name=LEADSSS
#MODULE LOAD 
module load gcc
module load python/3.8
conda init bash
conda activate ray

NODES=($(scontrol show hostnames $SLURM_NODELIST))

#RUN SCRIPT
for i in {0..4}; do
# srun --exclusive -N1 -n1 -w ${NODES[0]} --output=Easy_${i}.txt \
# python3 run.py --env_class maze --env_name Easy --N 75 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 \
# --lambda_e 0.1 --lambda_e_archive 0.1 --lambda_o 0.2 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 5e-4 --n_episode 16 \
# --n_update 16 --n_sgd_clearning 1024 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_1 --wandb_entity lds_algorithm &
# srun --exclusive -N1 -n1 -w ${NODES[1]} --output=Ur_${i}.txt \
# python3 run.py --env_class maze --env_name Ur --N 75 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 \
# --lambda_e 0.1 --lambda_e_archive 0.1 --lambda_o 0.2 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 5e-4 --n_episode 16 \
# --n_update 16 --n_sgd_clearning 1024 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_1 --wandb_entity lds_algorithm &
# srun --exclusive -N1 -n1 -w ${NODES[2]} --output=Hard_${i}.txt \
# python3 run.py --env_class maze --env_name Hard --N 75 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 \
# --lambda_e 0.1 --lambda_e_archive 0.1 --lambda_o 0.2 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 5e-4 --n_episode 16 \
# --n_update 16 --n_sgd_clearning 1024 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_1 --wandb_entity lds_algorithm &
# srun --exclusive -N1 -n1 -w ${NODES[0]} --output=fetch_${i}.txt \
# python3 run.py --env_class fetch --env_name fetch --N 50 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 \
# --lambda_e 0.02 --lambda_e_archive 0.02 --lambda_o 0.5 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 5e-4 --n_episode 16 \
# --n_update 16 --n_sgd_clearning 1024 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_2 --wandb_entity lds_algorithm &

# srun --exclusive -N1 -n1 -w ${NODES[1]} --output=finger_${i}.txt  \
# python3 run.py --env_class finger --env_name finger --N 90 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 \
# --lambda_e 0.1 --lambda_e_archive 0.01 --lambda_o 0.05 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 1e-3 --n_episode 8 \
# --n_update 16 --n_sgd_clearning 2048 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_2 --wandb_entity lds_algorithm &

# srun --exclusive -N1 -n1 -w ${NODES[1]} --output=hopper_${i}.txt \
# python3 run.py --env_class hopper --env_name hopper --N 50 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 \
# --lambda_e 0.1 --lambda_e_archive 0.01 --lambda_o 0.05 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 1e-3 --n_episode 8 \
# --n_update 16 --n_sgd_clearning 2048 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 1536 --batch_size_loss 1536 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_1 --wandb_entity lds_algorithm &

# srun --exclusive -N1 -n1 -w ${NODES[2]} --output=hand_${i}.txt \
# python3 run.py --env_class hand --env_name hand --N 21 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 \
# --lambda_e 0.005 --lambda_e_archive 0.001 --lambda_o 0.05 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 1e-3 --n_episode 16 \
# --n_update 32 --n_sgd_clearning 1024 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 1536 --batch_size_loss 1536 \
# --seed $i --device cpu --n_env 6 --wandb_project_name LDS_1 --wandb_entity lds_algorithm &

srun --exclusive -N1 -n1 -w ${NODES[0]} --output=fetch_${i}.txt \
python3 run.py --env_class fetch --env_name fetch --N 50 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 --featurize True \
--lambda_e 0.02 --lambda_e_archive 0.02 --lambda_o 0.5 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 5e-4 --n_episode 16 \
--n_update 16 --n_sgd_clearning 1024 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
--seed $i --device cpu --n_env 6 --n_c_r 128 --wandb_project_name LDS_3 --wandb_entity lds_algorithm &

srun --exclusive -N1 -n1 -w ${NODES[1]} --output=finger_${i}.txt \
python3 run.py --env_class finger --env_name finger --N 90 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 --featurize True \
--lambda_e 0.05 --lambda_e_archive 0.01 --lambda_o 0.05 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 1e-3 --n_episode 8 \
--n_update 16 --n_sgd_clearning 2048 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
--seed $i --device cpu --n_env 6 --n_c_r 128 --wandb_project_name LDS_3 --wandb_entity lds_algorithm &

srun --exclusive -N1 -n1 -w ${NODES[2]} --output=fetch_push_${i}.txt \
python3 run.py --env_class fetch_push --env_name fetch_push --N 84 --sigma 0.01 --n_skill 6 --z_dim 20 --lambda_d 0.5 --beta 1e5 --featurize True \
--lambda_e 0.005 --lambda_e_archive 0.005 --lambda_o 0.05 --gamma 0.95 --lambda_clearning 0.5 --lr_theta 5e-4 --lr_skill 5e-4 --lr_c 1e-3 --n_episode 8 \
--n_update 16 --n_sgd_clearning 2048 --n_sgd_discriminable_loss 16 --n_sgd_entropy_loss 16 --n_archive 1 --batch_size_clearning 3072 --batch_size_loss 3072 \
--seed $i --device cpu --n_env 6 --n_c_r 256 --wandb_project_name LDS_3 --wandb_entity lds_algorithm &
done
wait