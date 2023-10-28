#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=soc-gpu-np
module load miniconda3

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/uufs/chpc.utah.edu/sys/installdir/miniconda3/4.3.31/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/uufs/chpc.utah.edu/sys/installdir/miniconda3/4.3.31/etc/profile.d/conda.sh" ]; then
        . "/uufs/chpc.utah.edu/sys/installdir/miniconda3/4.3.31/etc/profile.d/conda.sh"
    else
        export PATH="/uufs/chpc.utah.edu/sys/installdir/miniconda3/4.3.31/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate jinghuang
cd /uufs/chpc.utah.edu/common/home/u1498392/wsi0.3125_exp/dev/resnet18layernormnl/4
python3 run.py 4
