


python ./vicuna/vicuna_worker.py \
    --name lmsys/vicuna-7b-v1.5-16k \
    --model_path lmsys/vicuna-7b-v1.5-16k \
    --controller_address http://127.0.0.1:21601 \
    $@

