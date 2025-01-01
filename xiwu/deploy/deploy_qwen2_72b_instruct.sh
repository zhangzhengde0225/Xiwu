export CUDA_VISIBLE_DEVICES=2,3


python ./qwen/qwen2_worker.py \
    --name Qwen2-72B-Instruct \
    --model_path /data1/sqr/Models/Qwen2-72B-Instruct \
    --gpus 2,3 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    --num_gpus 2 \
    $@

