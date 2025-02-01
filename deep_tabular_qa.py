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
import gc
import re
import json
import openai

from transformers import AutoModelForCausalLM, AutoTokenizer, CodeAgent
from datasets import Dataset, load_dataset

from databench_eval import Runner, Evaluator, utils
from databench_eval.utils import load_sample


from transformers import set_seed

set_seed(82)


def test_load_table(name, base_path='/home/dsartiano/semeval_2025_task_8/competition'):
    #f"hf://datasets/cardiffnlp/databench/data/{name}/all.parquet"
    return pd.read_parquet(f'{base_path}/{name}/all.parquet')
    #return pd.read_parquet()


def test_load_sample(name, base_path='/home/dsartiano/semeval_2025_task_8/competition'):
    #f"hf://datasets/cardiffnlp/databench/data/{name}/sample.parquet"
    return pd.read_parquet(f'{base_path}/{name}/sample.parquet')

class MyRunner(Runner):
    def process_prompt(self, prompts, datasets):
        raw_responses = self.model_call(prompts)
        responses = [
            self.postprocess(response=raw_response, dataset=dataset, prompt=prompt, row=self.qa[i+len(self.responses)])
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
        

    @classmethod
    def old_cleanup(cls, df, code):
        """ patch: drop extra lines """
        code = code.replace("<|EOT|>", "")
        lines = code.split('\n')
        extra_code = None
        for i,line in enumerate(lines):
            if 'def answer' in line:
                continue
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
    

    @classmethod
    def cleanup(cls, df, code):
        """Extract valid Python code, ensuring return statements are inside a function and removing extra non-code content."""
        code = code.replace("<|EOT|>", "").strip()
        lines = code.split("\n")

        python_code = []
        return_found = False  # Track if we have encountered a return statement

        for line in lines:
            stripped_line = line.strip()

            # Detect return statement
            if stripped_line.startswith("return "):
                return_found = True

            # Stop processing if we detect non-Python text after return
            if return_found and not stripped_line.startswith(("def ", "return ", "    ", "\t")):
                break  # Stop if we encounter metadata or unrelated text after return

            python_code.append(line)

        # If no function is found, wrap it inside `def answer(df):`
        if not any(line.lstrip().startswith("def ") for line in python_code):
            python_code.insert(0, "def answer(df):")
            python_code = ["    " + line if i > 0 else line for i, line in enumerate(python_code)]  # Indent body

        # Fix column names to match DataFrame
        columns = df.columns.to_list()
        for i, line in enumerate(python_code):
            matches = re.findall(r"'(\w+)'", line)
            for name in matches:
                if name not in columns:
                    for col in columns:
                        if col.startswith(f"{name}<"):
                            python_code[i] = python_code[i].replace(f"'{name}'", f"'{col}'")
                            break

        return '\n'.join(python_code)

    @classmethod
    def old_get_exec_string(cls, df, response):
        instruction = cls.cleanup(df, response)

        instructions = [el.strip() for el in instruction.split('\n')]
        if 'def answer' not in instruction:
            instructions.insert(0, "def answer(df):")
        
        exec_string = '\n'.join(['\t' + el if i > 0 else el for i, el in enumerate(instructions)])
        exec_string += "\nans = answer(df)"
        return exec_string
    
    @classmethod
    def get_exec_string(cls, df, response):
        """Prepare the cleaned function string for execution."""
        instruction = cls.cleanup(df, response)

        # Normalize indentation
        instructions = [line.rstrip() for line in instruction.split("\n") if line.strip()]
        min_indent = min((len(line) - len(line.lstrip())) for line in instructions if line.lstrip())

        exec_string = '\n'.join([line[min_indent:] for line in instructions])
        exec_string += "\nans = answer(df)"

        return exec_string
    
    def execute(self, exec_string, df):
        local_vars = {"df": df, "pd": pd, "np": np}
        exec(exec_string, local_vars)
        ans = local_vars['ans']
        
        if isinstance(ans, pd.Series):
            ans = ans.tolist()
        elif isinstance(ans, pd.DataFrame):
            ans = ans.iloc[:, 0].tolist()

        value = ans.split('\n')[0] if '\n' in str(ans) else ans
        return value

    def prompt_generator(self, row: dict) -> str:
        """ IMPORTANT: 
        **Only the question and dataset keys will be available during the actual competition**.
        You can, however, try to predict the answer type or columns used
        with another modeling task if that helps, then use them here.
        """
        dataset = row["dataset"]
        question = row["question"]
        if self.config.get('test', None):
            df = test_load_sample(dataset)
        else:
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
            value = None
            df = loader(dataset)
            global ans

            exec_string = ''

            lead = """
def answer(df):
    return """
            
            if 'model_chat_template' in self.config and self.config['model_chat_template']:
                lead = """
def answer(df): """
                instruction = response.strip()
                if 'def answer' in instruction:
                    lead = ''
                if len(instruction.split('\n')) > 1:    
                    instructions = [i.strip() for i in instruction.split('\n')]
                    if 'return' not in instruction:
                        instructions[-1] = f'return {instructions[-1]}'
                    instructions = ['\t'+i if 'def answer' not in i else i for i in instructions]
                    instruction = '\n'.join(instructions)
                    lead += '\n'
                else:
                    if 'return' not in instruction:
                        instruction = f'return {instruction}'
            elif 'batch_decode' in self.config and self.config['batch_decode']:
                exec_string = self.get_exec_string(df, response)
            else:
                instruction = response.split("return")[1].split("\n")[0].strip().replace("[end of text]", "")

            if not exec_string:
                exec_string = (
                    lead
                    + instruction
                    + "\nans = answer(df)"
                )

            value = self.execute(exec_string, df)

            if self.config.get('DUMP', False):
                with open(os.path.join(self.config['experiment_dir'], 'dump.json'), 'a') as dump_file:
                    print(json.dumps({'prompt': prompt, 'exec_string': exec_string, 'response': response, 'answer': value, 'truth': row}, default=str), file=dump_file)
                return

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
                    print(value)
                    print('|row')
                    print(row)
            # else:
            #     print('*'*80)
            #     print('|prompt')
            #     print(prompt)
            #     print('|dataset', dataset)
            #     print('|exec_string')
            #     print(exec_string)
            #     print('|response')
            #     print(response)
            #     print('|--> answer')
            #     print(value)
            #     print('|row')
            #     print(row)

            return value
        except Exception as e:
            print('*'*80)
            print(f'Error {e}')
            print('|prompt')
            print(prompt)
            print('|dataset', dataset)
            print('|response')
            print(response)
            print('|exec_string')
            print(exec_string)
            v_traceback = traceback.format_exc()
            print(v_traceback)

            if self.config.get('DUMP', False):
                with open(os.path.join(self.config['experiment_dir'], 'dump.json'), 'a') as dump_file:
                    print(json.dumps({'prompt': prompt, 'exec_string': exec_string, 'response': response, 'answer': value, 'truth': row}, default=str), file=dump_file)
                return

            if self.config.get('autofix', None):
                prompt_description = self.config.get('autofix', 'Provide a revised version of the function.')
                enriched_prompt = f'''
{prompt.rsplit('Table columns:', 1)[0]}

Given the following data:

Table columns: {df.columns.to_list()}
Dataframe: {df.head().to_json(orient='records')}
Question: {row["question"]}

The execution of this function:
{exec_string}

rises this Exception:

{v_traceback}

{prompt_description}

Table columns: {df.columns.to_list()}
Dataframe: {df.head().to_json(orient='records')}
Question: {row["question"]}
Function:
def answer(df: pd.DataFrame):
                '''
                response = self.generate(enriched_prompt).strip()
                exec_string = self.get_exec_string(df, response)
                
                print('|autofix prompt')
                print(enriched_prompt)
                print('|autofix response')
                print(response)
                print('|autofix exec_string')
                print(exec_string)
                try:
                    value = self.execute(exec_string, df)
                except Exception as e:
                    print('|autofix error', e)
                    print(traceback.format_exc())
                    return f"__CODE_ERROR__: {e}.".replace('\n', '\\n')
                
                print('|autofix value', value)
                return value

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
        if self.config.get('test', None):
            load_data_fun = test_load_table if not lite else test_load_sample
        else:
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


class ChatGPTDeepTabQA(DeepTabQA):
    def __init__(self, config):
        self.config = config
        self.client = openai.OpenAI(api_key=self.config['OPENAI_API_KEY'])

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=[
                {"role": "developer", "content": "You are a pandas code generator."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return completion.choices[0].message.content.replace('```python', '').replace('```', '').strip()
    
    @classmethod
    def get_exec_string(cls, df, response):
        return response + "\nans = answer(df)"

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
    if configuration.get('openai', False):
        deep_tab_qa = ChatGPTDeepTabQA(config=configuration)
    else:
        deep_tab_qa = DeepTabQA(config=configuration)
    
    perform_lite = configuration.get('lite', True)

    limit = configuration.get('limit', None)

    split = configuration.get('split', 'dev')

    batch_size = configuration.get('batch_size', 20)

    test = configuration.get('test', None)

    if test:
        if limit is not None:
            qa = load_dataset('/home/dsartiano/semeval_2025_task_8/competition')['test'].select(range(30))
        else:
            qa = load_dataset('/home/dsartiano/semeval_2025_task_8/competition')['test']
    elif limit is None:
        qa = utils.load_qa(name="semeval", split=split, num_proc=32)
    else:
        qa = utils.load_qa(name="semeval", split=split, num_proc=32).select(range(30))
    
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

    if not test:
        evaluator = Evaluator(qa=qa)
        deep_tab_qa.print_evaluation(evaluator=evaluator, responses=responses, responses_lite=responses_lite)

    deep_tab_qa.create_zip(file_predictions, file_predictions_lite)

if __name__ == '__main__':
    main()