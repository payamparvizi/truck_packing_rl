#!/bin/bash

#SBATCH --job-name=pack_rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --account=def-dspinell
#SBATCH --time=20:00:00
#SBATCH --mail-user=pparv056@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/payamp/
#SBATCH --array=0-80

module load StdEnv/2023
module load python/3.11
module load mujoco python
source /home/payamp/ENV_pack/bin/activate

# . $CONDA_ROOT/etc/profile.d/conda.sh

export WANDB_SERVICE_WAIT=90
export JAX_PLATFORMS=cpu

# parameters:
seeds=(1 2 3)
lrs=(1e-4 3e-4 1e-3)
eps_clips=(0.1 0.2 0.3)
ent_coefs=(0.0 0.001 0.005)

params=()
for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    for eps in "${eps_clips[@]}"; do
      for ent in "${ent_coefs[@]}"; do
        params+=("$seed $lr $eps $ent")
      done
    done
  done
done

param_set=${params[$SLURM_ARRAY_TASK_ID]}
read seed lr eps ent <<< "$param_set"

echo "Running with:"
echo "  seed=$seed"
echo "  lr=$lr"
echo "  eps_clip=$eps"
echo "  ent_coef=$ent"

python run_code.py \
  --seed "$seed" \
  --lr "$lr" \
  --eps_clip "$eps" \
  --ent_coef "$ent"
