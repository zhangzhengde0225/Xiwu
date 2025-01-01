export CUDA_VISIBLE_DEVICES=2


# python ./vicuna/vicuna_worker.py \
#     --name lmsys/vicuna-7b-v1.5-16k \
#     --model_path lmsys/vicuna-7b-v1.5-16k \
#     --gpus 2 \
#     --controller_address http://aiapi.ihep.ac.cn:42901 \
#     $@


python ./vicuna/xiwu_13b_20230509_worker.py \
    --name xiwu-13b-20230509 \
    --model_path /data1/sqr/Models/vicuna-13b-20230509 \
    --gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

