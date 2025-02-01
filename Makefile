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

OPENAI_API_KEY=

fine-tuning/chatpgt_dataset/dataset_gpt4o.json:
	OPENAI_API_KEY=$(OPENAI_API_KEY) $(PYTHON) fine-tuning/dataset_openai.py -model gpt-4o < fine-tuning/chatpgt_dataset/dump.json > $@

fine-tuning/chatpgt_dataset/dataset_gpt4o_postproc.json: fine-tuning/chatpgt_dataset/dataset_gpt4o.json
	$(PYTHON) fine-tuning/dataset_openai.py -post-processing < $< > $@

fine-tuning/chatpgt_dataset/dataset_gpt4o_truths.json: fine-tuning/chatpgt_dataset/dataset_gpt4o.json
	$(PYTHON) fine-tuning/dataset_openai.py -filter-truths < $< > $@

fine-tuning/chatpgt_dataset/dataset_gpt4o_truths_postproc.json: fine-tuning/chatpgt_dataset/dataset_gpt4o_truths.json
	$(PYTHON) fine-tuning/dataset_openai.py -post-processing < $< > $@


# dev

fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o.json:
	OPENAI_API_KEY=$(OPENAI_API_KEY) $(PYTHON) fine-tuning/dataset_openai.py -model gpt-4o < fine-tuning/chatpgt_dataset_dev/dump.json > $@

# In [2]: s = set()
#    ...: for i, l in enumerate(open('fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o.json')):
#    ...:     d = json.loads(l.strip())
#    ...:     if d['prompt'] in s:
#    ...:         continue
#    ...:     s.add(d['prompt'])
#    ...:     with open('fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o_unique.json', 'a') as fout:
#    ...:         print(l.strip(), file=fout)

fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o_postproc.json: fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o.json
	$(PYTHON) fine-tuning/dataset_openai.py -post-processing < $< > $@

fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o_truths.json: fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o_unique.json
	$(PYTHON) fine-tuning/dataset_openai.py -filter-truths -split dev < $< > $@

fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o_truths_postproc.json: fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o_truths.json
	$(PYTHON) fine-tuning/dataset_openai.py -post-processing < $< > $@

# dev+train
fine-tuning/chatpgt_dataset_dev/dataset_dev_train_gpt4o_truths_postproc.json:
	cat fine-tuning/chatpgt_dataset/dataset_gpt4o_truths_postproc.json fine-tuning/chatpgt_dataset_dev/dataset_dev_gpt4o_truths_postproc.json > $@

# KTO
fine-tuning/kto_dataset/wrong_answers.json:
	$(PYTHON) fine-tuning/dataset_kto.py < fine-tuning/kto_dataset/dump.json > $@

fine-tuning/KTO/dataset_deepseek.json:
	$(PYTHON) deep_tabular_qa_GA.py -dump > !$


first_submission:
	mkdir -p $@
	cp -r experiments/daniele_test_autofix_prompt2_dev_train $@/
	cp $@/daniele_test_autofix_prompt2_dev_train/predictions.txt $@/
	cp $@/daniele_test_autofix_prompt2_dev_train/predictions_lite.txt $@/

second_submission:
	mkdir -p $@
	cp -r experiments/GA_test_autofix_dev_train/ $@/
	cp $@/GA_test_autofix_dev_train/test-predictions.2.txt $@/predictions.txt
	cp $@/GA_test_autofix_dev_train/test-predictions_lite.2.txt $@/predictions_lite.txt

third_submission:
	mkdir -p $@
	cp -r experiments/GA_test_autofix_dev_train/ $@/
	cp $@/GA_test_autofix_dev_train/test-predictions.txt $@/predictions.txt
	cp $@/GA_test_autofix_dev_train/test-predictions_lite.txt $@/predictions_lite.txt
