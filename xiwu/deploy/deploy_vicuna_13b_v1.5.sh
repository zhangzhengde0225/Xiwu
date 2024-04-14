


python ./vicuna/vicuna_worker.py \
    --name lmsys/vicuna-13b-v1.5 \
    --model_path lmsys/vicuna-13b-v1.5 \
    --gpus 3 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

