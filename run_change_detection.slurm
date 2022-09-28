#!/bin/bash
#SBATCH --job-name="Multi_3Chnl_2016_2017_Same"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=100gb
#SBATCH --time=10-00:00:00
#SBATCH --partition=gpu 
#SBATCH --account=generalGrp
#SBATCH --output=my_job-%j.o
#SBATCH --error=my_error-%j.e
#SBATCH --mail-user=langefd@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1

module load rhel7/gpu/cuda/10.2
module load rhel7/gpu/cudnn/7.6.5
module load compilers/gcc/9.3.0


PYTHONPATH=/deac/csc/paucaGrp/camalas/software/python/3.7.11

PATH=$PYTHONPATH/bin:$PATH
PATH=$PYTHONPATH/include:$PATH
PATH=$PYTHONPATH/lib:$PATH
LD_LIBRARY_PATH=$PYTHONPATH/lib:$LD_LIBRARY_PATH

cd /deac/csc/paucaGrp/camalas/

source /deac/csc/paucaGrp/camalas/software/tensorflow_2.1/bin/activate

cd /deac/csc/paucaGrp/change-detection/years/Multi_3Chnl_2016_2017_Same

python3 -u train_model.py -d "/deac/csc/paucaGrp/change-detection/data/Original_data/2016_2017/" -r "multiclass" -c 3 -t1 "2016-09-17" -t2 "2017-10-17"> output_multi_3Chnl_2016_2017.txt

slurm_mem_report

exit 0