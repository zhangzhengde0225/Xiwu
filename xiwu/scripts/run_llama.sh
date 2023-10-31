#!/bin/bash
#SBATCH --job-name=l713b-v2
#SBATCH --mem=240GB
#SBATCH --gres=gpu:2
#SBATCH --partition=AI4HEP
#SBATCH --output=l13b-v2_%j.txt
#SBATCH --nodes=1
date

source ~/.bashrc
conda activate xiwu

#python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/vicuna_worker.py --model_path /dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/llama/llama-7b --name hepai/llama-7B --test True

#python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/vicuna_worker.py --model_path /dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/llama/llama-13b --name hepai/llama-13B --test True

python /dg_workfs/Beijing-CC/zdzhang/VSProjects/xiwu/xiwu/deploy/vicuna_worker.py --model_path /dg_workfs/Beijing-CC/zdzhang/DghpcData/weights/weights/llama2/llama-13b --name hepai/llama-13B-v2 --test True
