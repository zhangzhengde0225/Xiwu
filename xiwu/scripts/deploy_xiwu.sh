#!/bin/bash
#SBATCH --job-name=xiwu-13b
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --partition=AI4HEP
#SBATCH --output=xiwu_%j.txt
#SBATCH --nodes=1
date

source ~/.bashrc
conda activate xiwu

python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/xiwu_worker.py --test True