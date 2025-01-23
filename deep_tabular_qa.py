import os
import traceback
from typing import List
import argparse
import torch
import pandas as pd
import numpy as np
import zipfile
import jinja2
import yaml
import json
import gc
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, CodeAgent
from datasets import Dataset

from databench_eval import Runner, Evaluator, utils
from databench_eval.utils import load_sample


from transformers import set_seed

set_seed(82)
class MyRunner(Runner):
    def process_prompt(self, prompts, datasets):
        raw_responses = self.model_call(prompts)
        responses = [
            self.postprocess(response=raw_response, dataset=dataset, prompt=prompt, row=self.qa[i])
            for i, (raw_response, dataset, prompt) in enumerate(zip(raw_responses, datasets, prompts))
        ]
        self.prompts.extend(prompts)
        self.raw_responses.extend(raw_responses)
        self.responses.extend(responses)


class DeepTabQA:
    def __init__(self, config: dict):
        self.config = config
        
        print(self.config)
        if 'model_local' in self.config and self.config['model_local']:
            self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                torch_dtype="auto",
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                trust_remote_code=True,
                torch_dtype="auto",
            )
        self.model.cuda()
        
    def prompt_generator(self, row: dict) -> str:
        """ IMPORTANT: 
        **Only the question and dataset keys will be available during the actual competition**.
        You can, however, try to predict the answer type or columns used
        with another modeling task if that helps, then use them here.
        """
        dataset = row["dataset"]
        question = row["question"]
        df = load_sample(dataset)

        if 'prompt_template' in self.config:
            environment = jinja2.Environment(loader=jinja2.FileSystemLoader(self.config['experiment_dir']))
            template = environment.get_template(self.config['prompt_template'])
            s = template.render({'df': df, 'question': question})
        else:
            s = f'''
You are a pandas code generator. Your goal is to complete the function provided.
* You must not write any more code apart from that.
* You only have access to pandas and numpy.
* Pay attention to the type formatting .
* You cannot read files from disk.
* Don't write other functions. 


import pandas as pd
import numpy as np

def answer(df: pd.DataFrame):
    """Returns the answer to the question: {question}.
    The head of the dataframe, in json format, is: 
    {df.head().to_json(orient='records')}
    """
    df.columns = {list(df.columns)}
    return'''
        return s
    
    def postprocess(self, response: str, dataset: str, prompt: str, row: dict, loader):
        try:
            df = loader(dataset)
            global ans

            lead = """
def answer(df):
    return """
            
            def cleanup(code):
                """ patch: drop extra lines """
                code = code.replace("<|EOT|>", "")
                lines = code.split('\n')
                extra_code = None
                for i,line in enumerate(lines):
                    if not line.startswith(' '):
                        extra_code = i+1
                        break
                if extra_code:
                    lines = lines[:extra_code]
                # fix column names
                columns = df.columns.to_list()
                for i,line in enumerate(lines):
                    for m in re.finditer(r"'(\w+)'", line):
                        name = m.group(1)
                        if name not in columns:
                            for col in columns:
                                if col.startswith(f"{name}<"):
                                    lines[i] = lines[i].replace(f"'{name}'", f"'{col}'")
                                    break
                return '\n'.join(lines)

            if 'model_chat_template' in self.config and self.config['model_chat_template']:
                instruction = response.strip()
            elif 'batch_decode' in self.config and self.config['batch_decode']:
                lead = """
import pandas as pd
import numpy as np

def answer(df): """
                instruction = cleanup(response)
            else:
                instruction = response.split("return")[1].split("\n")[0].strip().replace("[end of text]", "")

            exec_string = (
                lead
                + instruction
                + "\nans = answer(df)"
            )

            if self.config.get('DUMP', False):
                with open(os.path.join(self.config['experiment_dir'], 'dump.json'), 'a') as dump_file:
                    print(json.dumps({'prompt': prompt, 'exec_string': exec_string}), file=dump_file)
                return

            local_vars = {"df": df, "pd": pd, "np": np}
            exec(exec_string, local_vars)
            ans = local_vars['ans']
            
            if isinstance(ans, pd.Series):
                ans = ans.tolist()
            elif isinstance(ans, pd.DataFrame):
                ans = ans.iloc[:, 0].tolist()

            value = ans.split('\n')[0] if '\n' in str(ans) else ans 

            if 'answer' in row:
                truth = row['sample_answer'] if loader == utils.load_sample else row['answer']
                if not self.evaluator.compare(value, truth, row['type']):
                    print('*'*80)
                    print('|prompt')
                    print(prompt)
                    print('|dataset', dataset)
                    print('|exec_string')
                    print(exec_string)
                    print('|response')
                    print(response)
                    print('|--> answer')
                    print(ans)
                    print('|row')
                    print(row)

            return value
        except Exception as e:
            print('*'*80)
            print(f'Error {e}')
            print('|response')
            print(response)
            print(traceback.format_exc())
            return f"__CODE_ERROR__: {e}.".replace('\n', '\\n')
    
    @staticmethod
    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def generate(self, prompt) -> str:

        max_new_tokens = self.config.get('max_new_tokens', 128)

        #escaped = prompt.replace('"', '\\"')
        if 'model_chat_template' in self.config and self.config['model_chat_template']:
            inputs = self.tokenizer.apply_chat_template([{'content': prompt, 'role': 'user'}], add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            tokens = self.model.generate(
                inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False, 
                num_return_sequences=1,
                temperature=0.2, 
                eos_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(tokens[0][len(inputs[0]):], skip_special_tokens=True)
        elif 'batch_decode' in self.config and self.config['batch_decode']:
            inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.model.device)
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            # strip the prompt
            return self.tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        else:
            inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.model.device)
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    def model_call(self, prompts: List) -> List:
        results = []
        for p in prompts:
            results.append(self.generate(p))
            self.flush()
        return results

    def get_runner(self, lite: bool, qa: Dataset, batch_size: int=10) -> Runner:
        load_data_fun = utils.load_table if not lite else utils.load_sample
        self.runner = MyRunner(
            model_call=self.model_call,
            prompt_generator=self.prompt_generator,
            postprocess=lambda response, dataset, prompt, row: self.postprocess(
                response=response, dataset=dataset, prompt=prompt, row=row, loader=load_data_fun
            ),
            qa=qa,
            batch_size=batch_size,
        )

        self.evaluator = Evaluator(qa=qa)

        return self.runner

    def print_evaluation(self, evaluator, responses, responses_lite):
        accuracy = evaluator.eval(responses)
        accuracy_lite = evaluator.eval(responses_lite, lite=True)
        with open(os.path.join(self.config['experiment_dir'], 'evaluation.txt'), 'w') as evalfile:
            print(yaml.dump(self.config), file=evalfile)
            print(f"DataBench accuracy is {accuracy}", file=evalfile)
            print(f"DataBench_lite accuracy is {accuracy_lite}", file=evalfile)
        print(f"DataBench accuracy is {accuracy}")
        print(f"DataBench_lite accuracy is {accuracy_lite}")

    def create_zip(self, file_predictions: str, file_predictions_lite: str):
        with zipfile.ZipFile(os.path.join(self.config['experiment_dir'], "Archive.zip"), "w") as zipf:
            zipf.write(file_predictions)
            if file_predictions_lite:
                zipf.write(file_predictions_lite)
        print(f"Created Archive.zip containing {file_predictions} and {file_predictions_lite}")


from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_huggingface import HuggingFacePipeline

class LangChainAgentDeepTabQA(DeepTabQA):
    def __init__(self, config: dict):
        self.config = config

        print(self.config)

        self.llm = HuggingFacePipeline.from_model_id(
            model_id=config['model_name'],
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 128,
                #"top_k": 50,
                "temperature": 0.2,
            },
        )

    def prompt_generator(self, row: dict) -> str:
        return row
    
    def model_call(self, rows: List):
        return rows

    def postprocess(self, row: dict, dataset: str, loader):
        agent_executor = create_pandas_dataframe_agent(
            self.llm,
            loader(dataset),
            #agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True
        )

        print(row)

        return agent_executor.invoke(row['question'])

def main():

    parser = argparse.ArgumentParser(prog='')
    parser.add_argument('config', help='A configuration file')

    args = parser.parse_args()

    with open(args.config) as fconfig:
        configuration = yaml.safe_load(fconfig)

    if not configuration['experiment_dir']:
        raise Exception('Set the experiment_dir in the config.yaml')
    
    if not configuration['description']:
        raise Exception('Set the description in the config.yaml')
    
    if 'class' in configuration and configuration['class'] == 'LangChainAgentDeepTabQA':
        deep_tab_qa = LangChainAgentDeepTabQA(config=configuration)
    else:
        deep_tab_qa = DeepTabQA(config=configuration)
    
    perform_lite = configuration.get('lite', True)

    limit = configuration.get('limit', None)

    split = configuration.get('split', 'dev')

    batch_size = configuration.get('batch_size', 20)

    if limit is None:
        qa = utils.load_qa(name="semeval", split=split, num_proc=32)
    else:
        qa = utils.load_qa(name="semeval", split=split, num_proc=32).select(range(10))
    
    print('dataset loaded')
    
    runner = deep_tab_qa.get_runner(lite=False, qa=qa, batch_size=batch_size)
    file_predictions = os.path.join(deep_tab_qa.config['experiment_dir'], "predictions.txt")
    responses = runner.run(save=file_predictions)

    if perform_lite:
        runner_lite = deep_tab_qa.get_runner(lite=True, qa=qa, batch_size=batch_size)
        file_predictions_lite = os.path.join(deep_tab_qa.config['experiment_dir'], "predictions_lite.txt")
        responses_lite = runner_lite.run(save=file_predictions_lite)
    else:
        responses_lite = []
        file_predictions_lite = None

    evaluator = Evaluator(qa=qa)

    deep_tab_qa.print_evaluation(evaluator=evaluator, responses=responses, responses_lite=responses_lite)
    deep_tab_qa.create_zip(file_predictions, file_predictions_lite)

if __name__ == '__main__':
    main()