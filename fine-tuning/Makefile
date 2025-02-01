MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
#MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Llama-8B

PROMPT = ../prompts/prompt_improved_one_line.txt

VERSION=15

#DATASET=fine-tuning-dataset-chat-template-v7.json
#DATASET=chatpgt_dataset/dataset_gpt4o_postproc.json
#DATASET=chatpgt_dataset/dataset_gpt4o_truths_postproc.json
DATASET=chatpgt_dataset_dev/dataset_dev_train_gpt4o_truths_postproc.json

DATASET_VERSION=dataset_dev_train_gpt4o_truths_postproc

CHAT_TEMPLATE=-model_chat_template

# default was 1000
MAX_STEPS=1000

%.json:
	python dataset_maker.py $@ "$(MODEL_NAME)" "$(PROMPT)" $(CHAT_TEMPLATE)

# lora + orpo
$(MODEL_NAME)-max_steps$(MAX_STEPS)-dataset_$(DATASET_VERSION)-$(VERSION)-lora-aligned-orpo:
	python orpo.py \
		--dataset_name $(DATASET) \
		--model_name_or_path=$(MODEL_NAME) \
		--per_device_train_batch_size 4 \
		--max_steps $(MAX_STEPS) \
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
		--lora_alpha=16 \
		--max_prompt_length=320

EPOCHS=20
#KTO_DATASET=chatpgt_dataset/dataset_gpt4o_truths_postproc.json
KTO_DATASET=../experiments/GA_09ft/train-dump.json
KTO_DATASET_VERSION=GA_09ft
# QLoRA + KTO
$(MODEL_NAME)-epochs$(EPOCHS)-dataset_$(KTO_DATASET_VERSION)-$(VERSION)-lora-aligned-kto:
	python kto.py \
		--dataset_name $(KTO_DATASET) \
		--model_name_or_path=$(MODEL_NAME) \
		--per_device_train_batch_size 2 \
		--num_train_epochs $(EPOCHS) \
		--learning_rate 5e-7 \
		--lr_scheduler_type=cosine \
		--gradient_accumulation_steps 1 \
		--logging_steps 10 \
		--eval_steps 500 \
		--output_dir="$@" \
		--warmup_ratio 0.1 \
		--bf16 \
		--logging_first_step \
		--use_peft \
		--load_in_4bit \
		--lora_target_modules=all-linear \
		--lora_r=16 \
		--lora_alpha=16 \
		--max_prompt_length=320

# full training with ORPO
$(MODEL_NAME)-lora-aligned-orpo-regular:
	python orpo.py \
		--dataset_name fine_tuning_dataset.json \
		--model_name_or_path=$(MODEL_NAME) \
		--per_device_train_batch_size 1 \
		--max_steps 1000 \
		--learning_rate 8e-6 \
		--gradient_accumulation_steps 1 \
		--logging_steps 10 \
		--eval_steps 500 \
		--output_dir="$@" \
		--warmup_steps 150 \
		--bf16 \
		--logging_first_step \
		--no_remove_unused_columns
