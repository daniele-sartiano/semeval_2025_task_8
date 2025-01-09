from typing import List
import argparse
import torch
import pandas as pd
import numpy as np
import zipfile

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from databench_eval import Runner, Evaluator, utils
from databench_eval.utils import load_sample


class DeepTabQA:
    def __init__(self, model_name: str='mistralai/Mistral-7B-Instruct-v0.3'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.model.cuda()

    @staticmethod
    def prompt_generator(row: dict) -> str:
        """ IMPORTANT: 
        **Only the question and dataset keys will be available during the actual competition**.
        You can, however, try to predict the answer type or columns used
        with another modeling task if that helps, then use them here.
        """
        dataset = row["dataset"]
        question = row["question"]
        df = load_sample(dataset)

        #pd.set_option('display.max_rows', None)
        #pd.set_option('display.max_columns', None)

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
    
    @staticmethod
    def postprocess(response: str, dataset: str, loader):
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
        
    def model_call(self, prompts: List) -> List:
        results = []
        for p in prompts:
            #escaped = p.replace('"', '\\"')
            inputs = self.tokenizer(p, return_tensors="pt").to(self.model.device)
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            answer = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            results.append(answer)
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

def main():

    #parser = argparse.ArgumentParser(prog='')
    #parser.add_argument('-o', "--opts",)   

    deep_tab_qa = DeepTabQA(model_name='mistralai/Mistral-7B-Instruct-v0.3')
    
    qa = utils.load_qa(name="qa")#.select(range(100))
    
    runner = deep_tab_qa.get_runner(lite=False, qa=qa, batch_size=100)
    runner_lite = deep_tab_qa.get_runner(lite=True, qa=qa, batch_size=100)

    evaluator = Evaluator(qa=qa)

    responses = runner.run(save="predictions.txt")
    responses_lite = runner_lite.run(save="predictions_lite.txt")

    print(f"DataBench accuracy is {evaluator.eval(responses)}")  # ~0.15
    print(f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}")  # ~0.07

    with zipfile.ZipFile("Archive.zip", "w") as zipf:
        zipf.write("predictions.txt")
        zipf.write("predictions_lite.txt")
    print("Created Archive.zip containing predictions.txt and predictions_lite.txt")

if __name__ == '__main__':
    main()