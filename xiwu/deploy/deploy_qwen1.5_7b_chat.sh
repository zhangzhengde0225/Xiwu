export CUDA_VISIBLE_DEVICES=3


python ./qwen/qwen_worker.py \
    --name Qwen1.5-7B-Chat \
#    --model_path /data1/sqr/Models/Qwen1.5-7B-Chat \
    --model_path /data1/sqr/Models/XW_qwen1.5 \
    --gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

