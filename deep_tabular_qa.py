import os
from typing import List
import argparse
import torch
import pandas as pd
import numpy as np
import zipfile
import jinja2
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, CodeAgent
from datasets import Dataset

from databench_eval import Runner, Evaluator, utils
from databench_eval.utils import load_sample


from transformers import set_seed

set_seed(82)

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
    
    def postprocess(self, response: str, dataset: str, loader):
        try:
            df = loader(dataset)
            global ans

            lead = """
def answer(df):
    return """
            exec_string = (
                lead
                + response.split("return")[1].split("\n")[0].strip().replace("[end of text]", "")
                + "\nans = answer(df)"
            )

            local_vars = {"df": df, "pd": pd, "np": np}
            exec(exec_string, local_vars)
            ans = local_vars['ans']
            
            print(response)
            print('--> answer')
            print(ans)

            if isinstance(ans, pd.Series):
                ans = ans.tolist()
            elif isinstance(ans, pd.DataFrame):
                ans = ans.iloc[:, 0].tolist()
            return ans.split('\n')[0] if '\n' in str(ans) else ans
        except Exception as e:
            print(e)
            return f"__CODE_ERROR__: {e}"
        
    def generate(self, prompt) -> str:
        #escaped = p.replace('"', '\\"')
        inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.model.device)
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    def model_call(self, prompts: List) -> List:
        results = []
        for p in prompts:
            results.append(self.generate(p))
        return results

    def get_runner(self, lite: bool, qa: Dataset, batch_size: int=10) -> Runner:
        load_data_fun = utils.load_table if not lite else utils.load_sample
        runner = Runner(
            model_call=self.model_call,
            prompt_generator=self.prompt_generator,
            postprocess=lambda response, dataset: self.postprocess(
                response, dataset, load_data_fun
            ),
            qa=qa,
            batch_size=batch_size,
        )
        return runner

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
    
    qa = utils.load_qa(name="semeval", split="dev")
    #qa = utils.load_qa(name="qa").select(range(10))
    
    runner = deep_tab_qa.get_runner(lite=False, qa=qa, batch_size=1000)
    runner_lite = deep_tab_qa.get_runner(lite=True, qa=qa, batch_size=1000)

    evaluator = Evaluator(qa=qa)

    file_predictions = os.path.join(deep_tab_qa.config['experiment_dir'], "predictions.txt")
    file_predictions_lite = os.path.join(deep_tab_qa.config['experiment_dir'], "predictions_lite.txt")

    responses = runner.run(save=file_predictions)
    responses_lite = runner_lite.run(save=file_predictions_lite)

    deep_tab_qa.print_evaluation(evaluator=evaluator, responses=responses, responses_lite=responses_lite)
    deep_tab_qa.create_zip(file_predictions, file_predictions_lite)

if __name__ == '__main__':
    main()