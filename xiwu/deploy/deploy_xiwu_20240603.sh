export CUDA_VISIBLE_DEVICES=2


python ./llama/xiwu_20240603_worker.py \
    --name Xiwu_20240603 \
    --model_path /data1/sqr/Models/Llama-8B-Instruct-train_idf_2e-5-2epoch \
    --gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

