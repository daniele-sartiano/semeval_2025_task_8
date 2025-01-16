import re
import json
import argparse
import pandas as pd
import numpy as np

import jinja2
from databench_eval import utils, Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

class DatasetMaker:
    def __init__(self, config: dict):
        self.config = config

        self.qa = utils.load_qa(name="semeval", split="train")

        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.model.cuda()


    def invoke_llm(self, prompt: str):
        inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.model.device)
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def create_rejected(self, question: str, instruction: str):

        prompt = f"""
Modify the following Python instruction to return an incorrect value for the question: '{question}'.
Create an alternative version with the main instruction altered, so that it returns an incorrect value.
The error can be simple or non-trivial, but must remain in one line. Only use pandas and numpy.
Do not include any additional text or explanations. Write just one.
Respond with only the modified code in the following format:

<code>{instruction}</code>
        """

        response = self.invoke_llm(prompt=prompt)
        answer = response.replace(prompt, '')
        
        print(answer)
        
        pattern = r"<code>(.*?)</code>"
        matches = re.findall(pattern, answer)
        for match in filter(lambda x: x.strip() != instruction.strip(), matches):
            yield match

    def generate(self):
        evaluator = Evaluator()
        for row in self.qa:
            # example {'question': 'Is the person with the highest net worth self-made?', 'answer': 'True', 'type': 'boolean', 'columns_used': "['finalWorth', 'selfMade']", 'column_types': "['number[uint32]', 'boolean']", 'sample_answer': 'False', 'dataset': '001_Forbes'}
            dataset = row["dataset"]
            question = row["question"]
            truth = row['answer']
            truth_lite = row['sample_answer']
            
            df = utils.load_table(dataset)
            df_lite = utils.load_sample(dataset)
            environment = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
            template = environment.get_template(self.config['prompt_template'])
            prompt = template.render({'df': df, 'question': question})

            llm_response = self.invoke_llm(prompt)
            # instruction contains something like df[df['selfMade'] == True]['personName'].iloc[0]
            instruction = llm_response.split("return")[1].split("\n")[0].strip().replace("[end of text]", "")
            
            global ans, ans_lite
            try:
                lead = """
def answer(df):
    return """
                exec_string = (
                    lead
                    + instruction
                    + "\nans = answer(df)"
                )

                local_vars = {"df": df, "pd": pd, "np": np}
                exec(exec_string, local_vars)
                ans = local_vars['ans']

                local_vars_lite = {"df": df_lite, "pd": pd, "np": np}
                exec(exec_string, local_vars_lite)
                ans_lite = local_vars_lite['ans']
                
                # evaluate gold vs generated
                if evaluator.compare(ans, truth, row['type']):
                    with open(self.config['fine_tuning_dataset'], 'a+') as fout:
                        for entry in set([json.dumps({'prompt': question, 'chosen': instruction, 'rejected': rej}) for rej in self.create_rejected(question, instruction=instruction)]):
                            print(entry, file=fout)
                # lite task
                if evaluator.compare(ans_lite, truth_lite, row['type']):
                    with open(self.config['fine_tuning_dataset_lite'], 'a+') as fout:
                        for entry in set([json.dumps({'prompt': question, 'chosen': instruction, 'rejected': rej}) for rej in self.create_rejected(question, instruction=instruction)]):
                            print(entry, file=fout)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'Error {e}')
            

def main():

    parser = argparse.ArgumentParser(prog='')
    parser.add_argument('action', choices=['dataset', 'fine-tuning'])

    args = parser.parse_args()
    if args.action == 'dataset':
        config = {
            'model_name': 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'prompt_template': 'prompts/prompt_improved_1.txt',
            'fine_tuning_dataset': 'fine_tuning_dataset.json',
            'fine_tuning_dataset_lite': 'fine_tuning_dataset_lite.json'
        }

        d = DatasetMaker(config=config)
        d.generate()
    elif args.action == 'fine-tuning':
        dataset = load_dataset('json', data_files='fine_tuning_dataset.json')
        dataset.save_to_disk('ft-dataset')

if __name__ == '__main__':
    main()

