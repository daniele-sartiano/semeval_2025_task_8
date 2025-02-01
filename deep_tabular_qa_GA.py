import os, sys
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
import re
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, CodeAgent
from datasets import Dataset, load_dataset

from databench_eval import Runner, Evaluator, utils
from typing import Callable, List, Optional
from tqdm import tqdm

from transformers import set_seed
set_seed(82) # 17 produces worse code

table_base_path = "hf://datasets/cardiffnlp/databench/data" # "./comnpetition" for test dataset

def load_table(name):
    global table_base_path
    return pd.read_parquet(f'{table_base_path}/{name}/all.parquet')


def load_sample(name):
    global table_base_path
    return pd.read_parquet(f'{table_base_path}/{name}/sample.parquet')

class MyRunner(Runner):
    def process_prompt(self, prompts: List[str], datasets: List[str], rows):
        """Handle a batch of prompts on the given datasets"""
        raw_responses = self.model_call(prompts)
        responses = [
            self.postprocess(response=raw_response, dataset=dataset, prompt=prompt, row=row)
            for raw_response, dataset, prompt, row in zip(raw_responses, datasets, prompts, rows)
        ]
        self.prompts.extend(prompts)
        self.raw_responses.extend(raw_responses)
        self.responses.extend(responses)

    def run(
        self,
        prompts: Optional[list[str]] = None,
        save: Optional[str] = None,
    ) -> List[str]:
        if prompts is not None:
            if len(prompts) != len(self.qa):
                raise ValueError("n_prompts != n_qa")

            for i in tqdm(range(0, len(prompts), self.batch_size)):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_datasets = self.qa[i : i + self.batch_size]["dataset"]
                self.process_prompt(batch_prompts, batch_datasets)
        else:
            if self.prompt_generator is None:
                raise ValueError("Generator must be provided if prompts are not.")
            for i in tqdm(range(0, len(self.qa), self.batch_size)):
                batch_rows = self.qa.select(
                    range(i, min(i + self.batch_size, len(self.qa)))
                )
                batch_prompts = [self.prompt_generator(row) for row in batch_rows]
                batch_datasets = [row["dataset"] for row in batch_rows]
                self.process_prompt(batch_prompts, batch_datasets, batch_rows)

        if save:
            self.save_responses(save)
        return self.responses


class DeepTabQA:
    def __init__(self, config: dict, qa: Dataset):
        self.config = config
        self.qa = qa
        self.dump_file = ''

        model_name = config['model_name']
        if self.config.get('model_local'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto",
            )
        self.model.cuda()
        
        self.evaluator = Evaluator(qa=qa)

    def prompt_generator(self, row: dict) -> str:
        """ IMPORTANT: 
        **Only the question and dataset keys will be available during the actual competition**.
        You can, however, try to predict the answer type or columns used
        with another modeling task if that helps, then use them here.
        """
        dataset = row["dataset"]
        question = row["question"]
        answer_type = row.get('type') # not present in test set

        df = load_sample(dataset) # just first 20 rows, enough for creating the prompt

        if 'prompt_template' in self.config:
            environment = jinja2.Environment(loader=jinja2.FileSystemLoader(self.config['experiment_dir']))
            template = environment.get_template(self.config['prompt_template'])
            prompt = template.render({'df': df, 'question': question})
        else:
            prompt = f'''
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
'''
        return prompt
    
    def postprocess(self, response: str, dataset: str, prompt: str, row: dict, lite: bool = False):
        loader = load_table if not lite else load_sample
        df = loader(dataset)
        answer = row.get('sample_answer') if lite else row.get('answer')
        answer_type = row.get('type')
        question = row["question"]

        def cleanup(code):
            """ extract plain code

            Responses to fix:

            RESPONSE:
            Function:
            def answer(df: pd.DataFrame):
            def answer(df: pd.DataFrame):
                return df[df['Organization'] == 'IBM']['JobRole'].nunique()

            RESPONSE:
            The function:
            def answer(df: pd.DataFrame):
                return df['ratings'].apply(lambda x: pd.Series(eval(x))[1]).min() == 1

            RESPONSE:
            The function should:

            1. Filter the dataframe to include only rows where the author field is not null.
            2. Extract the usernames from the author field and store them in a new dataframe.
            3. Filter the new dataframe to include only rows where the username is not null.
            4. Group the new dataframe by the username and count the number of reviews each user has written.
            5. Return the usernames of the top 4 users who have written the most reviews. If there are fewer than 4 users, return all usernames.


            RESPONSE:
            def answer(df: pd.DataFrame):
                def extract_rating(x: str, rating: str):
                    return eval(x)[rating]

                return df['ratings'].apply(lambda x: extract_rating(x, 'rooms')).mean()

            RESPONSE:
            Please note that the function should be written in pandas.

            The function should not use any external libraries.

            The function should be written in a single line.

            Function:
            def answer(df: pd.DataFrame):
                return df[df['author'].notnull() & df['author'].str.extract('username\s*:\s*(\w+)', flags=re.IGNORECA
            """
            # MarkDown
            m = re.findall(r"```python\n(.*?)```", code, re.DOTALL)
            if m:
                return m[-1]    # just last one, in conversation.

            # Check for indented code:
            pattern = r"\n?((?:[ \t]+.*\n?)+)"
            m = re.match(pattern, code)
            if m:
                body = m.group(1)
                if not re.search(r'\sreturn ', body):
                    # sometimes the LLM loops
                    code = 'def answer(df): return None\n'
                else:
                    code = f'def answer(df):\n{body}\n'
            else:
                # Look for a full definition
                # Define a regex pattern to match function definitions
                pattern = r"def answer\((.*?)\):\n?((?:[ \t]+.*\n?)+)"

                # Search for the pattern in the response
                match = re.search(pattern, code)
                if match:
                    params = match.group(1)
                    body = match.group(2)
                    code = f'def answer(df):\n{body}\n'
                else:
                    # Just the expression to return
                    if code.startswith('\n'):
                        code = code[1:]
                    if not re.search(r'\sreturn ', code):
                        code = 'def answer(df): return None\n'
            if not code:
                return "def answer(df): return None\n"

            # # patch: drop extra lines """
            # code = code.replace("<|EOT|>", "")
            
            # lines = code.split('\n')
            # extra_code = None
            # for i,line in enumerate(lines):
            #     # skip def answer if present
            #     # if i == 0 and line.startswith('def answer('):
            #     #     continue
            #     if not line.startswith((' ', '\t')):
            #         extra_code = i+1
            #         break
            # if extra_code:
            #     lines = lines[:extra_code]
            # code = '\n'.join(lines)

            # fix column names
            columns = df.columns.to_list()
            for m in re.finditer(r"'(\w+)'", code):
                name = m.group(1)
                if name not in columns:
                    for col in columns:
                        if col.startswith(f"{name}<"):
                            code = code.replace(name, col)
                            break
                        # # misspelled name
                        # if col == name.lower():
                        #     code = code.replace(name, col)
                        #     break

            # if not code.startswith("def answer("):
            #     code = "def answer(df: pd.DataFrame):\n" + code

            # remove extra punct
            code = code.rstrip()
            if code.endswith(('.', ':')):
                code = code[:-1]
            code = code + '\n\n'
            
            return code

        code = cleanup(response)
            
        lead = """
import pandas as pd
import numpy as np
import re
"""
       
        exec_string = (
                lead
                + code
                + "\nans = answer(df)"
            )
            
        if self.config.get('DEBUG', False):
            print('PROMPT:', prompt)
            print('RESPONSE:', response)
            print('CODE:', exec_string)

        error_prompt = ''
        try:
            local_vars = {"df": df}
            exec(exec_string, local_vars)
            predict = local_vars['ans']
            
            if isinstance(predict, pd.Series):
                predict = predict.tolist()
            elif isinstance(predict, pd.DataFrame):
                predict = predict.iloc[:, 0].tolist()
            predict = str(predict).split('\n')[0] # keep only first if multiline answers

            # check answer type
            if (question.endswith('Answer True or False') or
                question.startswith(('Is', 'Are', 'Does'))) and predict not in ('False', 'True'):
                error_prompt = """
The previous function does not return a boolean.
Please provide a function that answers the previous question and returns either True or False.
"""
            elif question.endswith('Answer with a single category.') and predict.startswith('['):
                error_prompt = """
The previous function does not return a single value.
Please provide a function that answers the previous question and returns a single value.
"""

        except Exception as e:
            if self.config.get('DEBUG', False):
                print(f"__FIRST_ERROR__: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                exc_message = f"{type(e).__name__}: {e}" # get out of nested exception
                error_prompt = f"""
The previous function rises this Exception: {exc_message}.
Please provide a function that answers the previous question and avoids this error.
"""
        # Try fixing the code:
        if error_prompt:
            if prompt.endswith('\n<think>\n'): # R1
                code = f"```python\n{code}\n```\n"
                prompt2 = (prompt[:-len('<think>\n')]
                           + code + error_prompt
                           + "<think>\n")
            else:
                prompt2 = prompt + response + error_prompt

            # invoke LLM again
            response = self.generate(prompt2)

            code = cleanup(response)
            
            exec_string = (
                lead
                + code
                + "\nans = answer(df)"
            )

            if self.config.get('DEBUG', False):
                print('2nd PROMPT:', prompt2)
                print('2nd RESPONSE:', response)
                print('2nd CODE:', exec_string)

            try:
                local_vars = {"df": df} # redundant?
                exec(exec_string, local_vars)
                predict = local_vars['ans']
            
                if isinstance(predict, pd.Series):
                    predict = predict.tolist()
                elif isinstance(predict, pd.DataFrame):
                    predict = predict.iloc[:, 0].tolist()
                predict = str(predict).split('\n')[0] # keep only first if multiline answers
            except Exception as e:
                predict = f"__CODE_ERROR__: {type(e).__name__}: {e}".replace(r'\n', '\\n')
                if self.config.get('DEBUG', False):
                    print(traceback.format_exc())
            
        if self.dump_file and answer is not None:
            label = self.evaluator.compare(predict, answer, answer_type)
            with open(self.dump_file, 'a') as dump_file:
                print(f"{json.dumps({'prompt': prompt, 'completion': response, 'label': label})}", file=dump_file)

        print('ANSWER:', predict)
        if answer != None:      # test set
            print('TRUTH:', answer)

        return predict
        
    def generate(self, prompt) -> str:

        max_new_tokens = self.config.get('max_new_tokens', 128)

        #escaped = prompt.replace('"', '\\"')
        inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.model.device)
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        torch.cuda.empty_cache() # free memory

        # strip the prompt
        return self.tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    def model_call(self, prompts: List) -> List:
        return [self.generate(p) for p in prompts]

    def get_runner(self, lite: bool, batch_size: int=10) -> Runner:
        runner = MyRunner(
            model_call=self.model_call,
            prompt_generator=self.prompt_generator,
            postprocess=lambda response, dataset, prompt, row: self.postprocess(
                response, dataset, prompt, row, lite
            ),
            qa=self.qa,
            batch_size=batch_size,
        )
        return runner

    def print_evaluation(self, responses, filename, lite=False):
        accuracy = self.evaluator.eval(responses, lite=lite)
        # Append Lite evaluation mto the same file:
        with open(filename, 'a' if lite else 'w') as evalfile:
            if lite:
                print(f"DataBench_lite accuracy is {accuracy}", file=evalfile)
                print(f"DataBench_lite accuracy is {accuracy}")
            else:
                print(yaml.dump(self.config), file=evalfile)
                print(f"DataBench accuracy is {accuracy}", file=evalfile)
                print(f"DataBench accuracy is {accuracy}")

    def create_zip(self, file_predictions: str, file_predictions_lite: str):
        with zipfile.ZipFile(os.path.join(self.config['experiment_dir'], "Archive.zip"), "w") as zipf:
            zipf.write(file_predictions)
            if file_predictions_lite:
                zipf.write(file_predictions_lite)
        print(f"Created Archive.zip containing {file_predictions} and {file_predictions_lite}")


from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_huggingface import HuggingFacePipeline

class LangChainAgentDeepTabQA(DeepTabQA):
    def __init__(self, config: dict, qa: Dataset):
        self.config = config
        self.qa = qa

        self.llm = HuggingFacePipeline.from_model_id(
            model_id=config['model_name'],
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 128,
                #"top_k": 50,
                "temperature": 0.2,
            },
        )

        self.evaluator = Evaluator(qa=qa)


    def prompt_generator(self, row: dict) -> str:
        return row
    
    def model_call(self, rows: List):
        return rows

    def postprocess(self, row: dict, dataset: str, prompt: str, lite: bool):
        loader = load_table if not lite else load_sample
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
    parser.add_argument('-dump', action='store_true', help="Dump prompt, code, truth")
    parser.add_argument('-split', default='dev', help="Which dataset split to use (dev or train)")
    parser.add_argument('-batch-size', type=int, help="Size of batches per iteration)")

    args = parser.parse_args()

    with open(args.config) as fconfig:
        configuration = yaml.safe_load(fconfig)
    if args.dump:
        configuration['DUMP'] = True # override

    split_prefix = args.split + '-'

    batch_size = configuration.get('batch_size', 20)
    if args.batch_size:
        batch_size = args.batch_size
    
    if not configuration['experiment_dir']:
        raise Exception('Set the experiment_dir in the config.yaml')
    
    if not configuration['description']:
        raise Exception('Set the description in the config.yaml')
    
    print(configuration, file=sys.stderr)

    global table_base_path
    if args.split == "test":
        table_base_path = "./competition"
        qa = load_dataset(table_base_path, split=args.split)
    else:
        table_base_path = "hf://datasets/cardiffnlp/databench/data"
        qa = load_dataset("cardiffnlp/databench", name="semeval", split=args.split)

    limit = configuration.get('limit', None)
    if limit:
        qa = qa.select(range(limit)) # DEBUG

    if 'class' in configuration and configuration['class'] == 'LangChainAgentDeepTabQA':
        deep_tab_qa = LangChainAgentDeepTabQA(config=configuration, qa=qa)
    else:
        deep_tab_qa = DeepTabQA(config=configuration, qa=qa)
    
    runner = deep_tab_qa.get_runner(lite=False, batch_size=batch_size)
    if configuration['DUMP']:
        deep_tab_qa.dump_file = os.path.join(configuration['experiment_dir'], split_prefix + 'dump.json')
        # remove it, since postprocess will append results for each batch
        if os.path.exists(deep_tab_qa.dump_file):
            os.remove(deep_tab_qa.dump_file)

    start_time = datetime.now()
    print("Start:", start_time.strftime("%H:%M:%S"), file=sys.stderr)

    file_predictions = os.path.join(deep_tab_qa.config['experiment_dir'], split_prefix + "predictions.txt")
    responses = runner.run(save=file_predictions)

    print("Elapsed:", datetime.now() - start_time, file=sys.stderr)

    if args.split != 'test':
        eval_file = os.path.join(configuration['experiment_dir'], split_prefix + 'evaluation.txt')
        deep_tab_qa.print_evaluation(responses, eval_file)

    if configuration.get('lite', True):
        runner_lite = deep_tab_qa.get_runner(lite=True, batch_size=batch_size)
        file_predictions_lite = os.path.join(deep_tab_qa.config['experiment_dir'], split_prefix + "predictions_lite.txt")
        responses_lite = runner_lite.run(save=file_predictions_lite)
        if args.split != 'test':
            deep_tab_qa.print_evaluation(responses_lite, eval_file, lite=True)

    print("Elapsed:", datetime.now() - start_time, file=sys.stderr)

    deep_tab_qa.create_zip(file_predictions, file_predictions_lite)

if __name__ == '__main__':
    main()
