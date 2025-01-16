MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct

fine_tuning_dataset.json:
	python dataset_maker.py dataset

$(MODEL_NAME)-lora-aligned-orpo:
	python orpo.py \
		--dataset_name fine_tuning_dataset.json \
		--model_name_or_path=$(MODEL_NAME) \
		--per_device_train_batch_size 4 \
		--max_steps 1000 \
		--learning_rate 8e-5 \
		--gradient_accumulation_steps 1 \
		--logging_steps 10 \
		--eval_steps 500 \
		--output_dir="$@" \
		--optim rmsprop \
		--warmup_steps 150 \
		--bf16 \
		--logging_first_step \
		--no_remove_unused_columns \
		--use_peft \
		--lora_r=16 \
		--lora_alpha=16