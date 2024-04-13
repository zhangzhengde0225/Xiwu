


python ./vicuna/vicuna_worker.py \
    --name lmsys/vicuna-7b \
    --model_path lmsys/vicuna-7b \
    --device 2 \
    --controller_address http://aiapi.ihep.ac.cn:42901 \
    $@

