export CUDA_VISIBLE_DEVICES=1


# python ./vicuna/vicuna_worker.py \
#     --name lmsys/vicuna-7b-v1.5-16k \
#     --model_path lmsys/vicuna-7b-v1.5-16k \
#     --gpus 2 \
#     --controller_address http://aiapi.ihep.ac.cn:42901 \
#     $@


python ./vicuna/vicuna_worker.py \
    --name lmsys/vicuna-7b-v1.5-16k \
    --model_path /data1/sqr/Models/vicuna-7b-v1.5-16k \
    --gpus 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

