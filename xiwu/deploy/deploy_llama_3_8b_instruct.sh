export CUDA_VISIBLE_DEVICES=0

model_dir="/aifs/user/data/zdzhang/models"

python ./llama/llama_worker.py \
    --name Meta/Llama3-8B \
    --model_path $model_dir/Llama-3-8B-Instruct \
    --gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

