export CUDA_VISIBLE_DEVICES=3


python -W ignore ./baichuan/baichuan_worker.py \
    --name Baichuan2-7B-Chat \
    --model_path /data1/sqr/Models/Baichuan2-7B-Chat \
    --gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

