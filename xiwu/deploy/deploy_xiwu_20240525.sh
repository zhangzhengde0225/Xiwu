export CUDA_VISIBLE_DEVICES=3


python ./llama/xiwu_20240525_worker.py \
    --name Xiwu_20240525 \
    --model_path /data1/sqr/Models/Llama-3-8B-Instruct-all-1epoch-1e-5-idf-1e-4 \
    --gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

