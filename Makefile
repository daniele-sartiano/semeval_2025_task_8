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