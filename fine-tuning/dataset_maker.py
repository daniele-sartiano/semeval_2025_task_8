import re
import json
import argparse
import pandas as pd
import numpy as np

import jinja2
from databench_eval import utils, Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import os

from transformers import set_seed

set_seed(82)

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
        if 'model_chat_template' in self.config and self.config['model_chat_template']:
            inputs = self.tokenizer.apply_chat_template([{'content': prompt, 'role': 'user'}], add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            tokens = self.model.generate(
                inputs, 
                max_new_tokens=128, 
                do_sample=True, 
                #top_k=50, 
                #top_p=0.95,
                #num_return_sequences=1, 
                eos_token_id=self.tokenizer.eos_token_id)
            
            print(len(tokens))
            return self.tokenizer.decode(tokens[0][len(inputs[0]):], skip_special_tokens=True)
        else:
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
Do not include any additional text or explanations. Write minimun 5 samples.
Respond with only the modified code in the following format:

<code>{instruction}</code>
        """

        response = self.invoke_llm(prompt=prompt)
        answer = response.replace(prompt, '')
        
        print('rejected')
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
            environment = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.abspath(self.config['prompt_template']).replace(os.path.basename(self.config['prompt_template']), '')))
            template = environment.get_template(os.path.basename(self.config['prompt_template']))
            prompt = template.render({'df': df, 'question': question})

            llm_response = self.invoke_llm(prompt)


            if 'model_chat_template' in self.config and self.config['model_chat_template']:
                instruction = llm_response.strip()
                if len(instruction.split('\n')) > 0:
                    instruction = '; '.join(instruction.split('\n'))
            else:
                # instruction contains something like df[df['selfMade'] == True]['personName'].iloc[0]
                instruction = llm_response.split("return")[1].split("\n")[0].strip().replace("[end of text]", "")
            
            print(f'row\n{row}')
            print(f'llm_response\n{llm_response}')
            print(f'instruction\n{instruction}')

            global ans, ans_lite
            try:
                lead = """
def answer(df):
    return """
                if 'return' in instruction:
                    lead = lead.replace('return ', '')
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
                        for entry in set([json.dumps({'prompt': prompt, 'chosen': instruction, 'rejected': rej}) for rej in self.create_rejected(question, instruction=instruction)]):
                            print(entry, file=fout)
                # lite task
                #if evaluator.compare(ans_lite, truth_lite, row['type']):
                #    with open(self.config['fine_tuning_dataset_lite'], 'a+') as fout:
                #        for entry in set([json.dumps({'prompt': prompt, 'chosen': instruction, 'rejected': rej}) for rej in self.create_rejected(question, instruction=instruction)]):
                #           print(entry, file=fout)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'Error {e}')
            

def main():

    parser = argparse.ArgumentParser(prog='')
    
    parser.add_argument('name')
    parser.add_argument('model')
    parser.add_argument('prompt')
    parser.add_argument('-model_chat_template', action='store_true')

    args = parser.parse_args()

    
    dataset_name = 'fine_tuning_dataset.json' if not args.name else args.name
    dataset_name_lite = f"{dataset_name.split('.json')[0]}_lite.json"

    config = {
        'model_name': args.model,
        'prompt_template': args.prompt,
        'fine_tuning_dataset': dataset_name,
        'fine_tuning_dataset_lite': dataset_name_lite
    }

    if args.model_chat_template:
        config['model_chat_template'] = True

    print(config)

    d = DatasetMaker(config=config)
    d.generate()
    
    #dataset = load_dataset('json', data_files='fine_tuning_dataset.json')
    #dataset.save_to_disk('ft-dataset')

if __name__ == '__main__':
    main()

