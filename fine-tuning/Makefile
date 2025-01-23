MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct

PROMPT = ../prompts/prompt_improved_one_line.txt

VERSION=10

#DATASET=fine-tuning-dataset-chat-template-v7.json
DATASET=chatpgt_dataset/dataset_gpt4o_postproc.json

DATASET_VERSION=chatgpt

CHAT_TEMPLATE=-model_chat_template

# default was 1000
MAX_STEPS=30000

%.json:
	python dataset_maker.py $@ "$(MODEL_NAME)" "$(PROMPT)" $(CHAT_TEMPLATE)

$(MODEL_NAME)-max_steps$(MAX_STEPS)-dataset_$(DATASET_VERSION)-$(VERSION)-lora-aligned-orpo:
	python orpo.py \
		--dataset_name $(DATASET) \
		--model_name_or_path=$(MODEL_NAME) \
		--per_device_train_batch_size 2 \
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
		--lora_alpha=16

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
