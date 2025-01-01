export CUDA_VISIBLE_DEVICES=0,1


python ./llama/llama_worker_262k.py \
    --name Meta/Llama3-8B-262k \
    --model_path /data1/sqr/Models/Llama-3-8B-Instruct-262k \
    --gpus 0,1 \
    --num_gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    --max_new_tokens 262144 \
    --temperature 0.3 \
    $@

