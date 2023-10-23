#!/bin/bash
#SBATCH --job-name=v7b-v1.5
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --partition=AI4HEP
#SBATCH --output=v7b-v1.5_%j.txt
#SBATCH --nodes=1
date

source ~/.bashrc
conda activate xiwu

python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/vicuna_worker.py --model_path /dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/vicuna/vicuna-7b-v1.5-16k --name hepai/vicuna-7B-v1.5 --test True

#python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/vicuna_worker.py --model_path /dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/vicuna/vicuna-7b --name hepai/vicuna-7B --test True

#python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/vicuna_worker.py --model_path /dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/vicuna/vicuna-13b --name hepai/vicuna-13B --test True

#python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/vicuna_worker.py --model_path /dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/vicuna/vicuna-33b --name hepai/vicuna-33B --test True