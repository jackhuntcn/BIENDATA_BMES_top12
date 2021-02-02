cd ..
CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs

TASK_NAME="bien"

mkdir -pv preds_dir

for conf in "ce_16_2e-05_4" "ce_16_2e-05_5" "ce_16_3e-05_3" "ce_16_3e-05_4" "ce_24_2e-05_3" "ce_24_3e-05_5" "lsr_16_2e-05_4" "lsr_16_2e-05_5" "lsr_16_3e-05_3" "lsr_16_3e-05_4" "lsr_24_2e-05_3" "lsr_24_3e-05_5"; do

	LOSS=$(awk -F'_' '{print $1}' <<< "${conf}")
	BS=$(awk -F'_' '{print $2}' <<< "${conf}")
	LR=$(awk -F'_' '{print $3}' <<< "${conf}")
	EPOCHS=$(awk -F'_' '{print $4}' <<< "${conf}")

	FILENAME="predict_${LOSS}_${BS}_${LR}_${EPOCHS}.json"

	echo "${FILENAME}"
	sleep 5

	python run_ner_span.py \
	  --model_type=bert \
	  --model_name_or_path=$BERT_BASE_DIR \
	  --task_name=$TASK_NAME \
	  --do_train \
	  --do_adv \
	  --do_predict \
	  --do_lower_case \
	  --loss_type=${LOSS} \
	  --data_dir=$DATA_DIR/${TASK_NAME}/ \
	  --train_max_seq_length=134 \
	  --per_gpu_train_batch_size=${BS} \
	  --learning_rate=${LR} \
	  --num_train_epochs=${EPOCHS} \
	  --logging_steps=448 \
	  --save_steps=448 \
	  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
	  --overwrite_output_dir \
	  --seed=42

	mv -v outputs/bien_output/bert/test_predict.json "preds_dir/${FILENAME}"

done

cd biendata
