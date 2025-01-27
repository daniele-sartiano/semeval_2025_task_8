PYTHON = .env/bin/python -u

EXPERIMENTS_DIR=experiments/daniele_01
PROMPT_TEMPLATE=prompts/prompt_baseline.txt

# make PROMPT_TEMPLATE=prompts/prompt_baseline.txt EXPERIMENTS_DIR=experiments/daniele_01 template
# edit the file experiments/daniele_01/config.yaml (in particular set the experiment_dir value)
# make EXPERIMENTS_DIR=experiments/daniele_01 run

template:
	mkdir -p $(EXPERIMENTS_DIR)
	make $(EXPERIMENTS_DIR)/prompt.txt
	make $(EXPERIMENTS_DIR)/config.yaml

$(EXPERIMENTS_DIR)/prompt.txt:
	cp $(PROMPT_TEMPLATE) $@

$(EXPERIMENTS_DIR)/config.yaml:
	cp default_config.yaml $@

run: $(EXPERIMENTS_DIR)
	$(PYTHON) deep_tabular_qa.py $(EXPERIMENTS_DIR)/config.yaml

OPENAI_API_KEY=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

fine-tuning/chatpgt_dataset/dataset_gpt4o.json:
	OPENAI_API_KEY=$(OPENAI_API_KEY) $(PYTHON) fine-tuning/dataset_openai.py -model gpt-4o < fine-tuning/chatpgt_dataset/dump.json > $@

fine-tuning/chatpgt_dataset/dataset_gpt4o_postproc.json: fine-tuning/chatpgt_dataset/dataset_gpt4o.json
	$(PYTHON) fine-tuning/dataset_openai.py -post-processing < $< > $@

fine-tuning/chatpgt_dataset/dataset_gpt4o_truths.json: fine-tuning/chatpgt_dataset/dataset_gpt4o.json
	$(PYTHON) fine-tuning/dataset_openai.py -filter-truths < $< > $@

fine-tuning/chatpgt_dataset/dataset_gpt4o_truths_postproc.json: fine-tuning/chatpgt_dataset/dataset_gpt4o_truths.json
	$(PYTHON) fine-tuning/dataset_openai.py -post-processing < $< > $@

fine-tuning/kto_dataset/wrong_answers.json:
	$(PYTHON) fine-tuning/dataset_kto.py < fine-tuning/kto_dataset/dump.json > $@

fine-tuning/KTO/dataset_deepseek.json:
	$(PYTHON) deep_tabular_qa_GA.py -dump > !$
