#!/usr/bin/env bash

model_name=iTransformer
TSFModel=TimeMoe
mkdir -p logs1

datasets=(
  # "ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 ETTh1"
  "ETTh2 ./dataset/ETT-small/ ETTh2.csv 7 ETTh2"
  # "ETTm1 ./dataset/ETT-small/ ETTm1.csv 7 ETTm1"
  # "ETTm2 ./dataset/ETT-small/ ETTm2.csv 7 ETTm2"
  # "Weather ./dataset/weather/ weather.csv 21 custom"
)

seq_lens=(512 1024)
pred_lens=(96  192)
d_models=(512 1024)

total_jobs=$((${#datasets[@]} * ${#seq_lens[@]}))
job_id=0

for ds_info in "${datasets[@]}"; do
  read -r ds_name root_path data_path enc_in data_flag <<< "$ds_info"

  for ((i=0;i<${#seq_lens[@]};i++)); do
    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}
    d_model=${d_models[i]}
    job_id=$((job_id+1))

    model_id="${ds_name}_${seq_len}_${pred_len}_${model_name}_${TSFModel}"
    log_file="logs1/${model_id}.log"
    echo "==== [${job_id}/${total_jobs}]  ${model_id}  seq=${seq_len} pred=${pred_len} d_model=${d_model} ===="
    echo "日志 -> $log_file"

    python -u run.py \
      --task_name Exp_DistilTS \
      --is_training 1 \
      --pretrained_path ../TimeMoE-50M \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "$model_id" \
      --model "$model_name" \
      --data "$data_flag" \
      --features M \
      --seq_len "$seq_len" \
      --label_len 0 \
      --pred_len "$pred_len" \
      --d_model "$d_model" \
      --d_ff 2048 \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in "$enc_in" \
      --dec_in "$enc_in" \
      --c_out "$enc_in" \
      --des 'Exp' \
      --vt_loss 1 \
      --train_epochs 2 \
      --TSFModel $TSFModel \
      --itr 1 > "$log_file" 2>&1

  done
done