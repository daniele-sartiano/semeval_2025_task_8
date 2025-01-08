import pandas as pd
import numpy as np
import subprocess
import shlex
import zipfile
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from databench_eval.utils import load_sample

from datasets import Dataset
from databench_eval import Runner, Evaluator, utils


def call_my_model(prompts):
    results = []
    #tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-3b", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     "stabilityai/stable-code-3b",
    #     trust_remote_code=True,
    #     torch_dtype="auto",
    # )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.cuda()

    for p in prompts:
        escaped = p.replace('"', '\\"')
        inputs = tokenizer(escaped, return_tensors="pt").to(model.device)
        tokens = model.generate(
            **inputs,
            max_new_tokens=48,
            temperature=0.2,
            do_sample=True,
        )
        results.append(tokenizer.decode(tokens[0], skip_special_tokens=True))
    return results

def example_generator(row: dict) -> str:
    """ IMPORTANT: 
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    dataset = row["dataset"]
    question = row["question"]
    df = load_sample(dataset)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    return f'''
You are a pandas code generator. Your goal is to complete the function provided.
* You must not write any more code apart from that.
* You only have access to pandas and numpy.
* Pay attention to the type formatting .
* You cannot read files from disk.


import pandas as pd
import numpy as np

def answer(df: pd.DataFrame):
    """Returns the answer to the question: {question}.
       The head of the dataframe is: 
       {df.head()}
    """
    df.columns = {list(df.columns)}
    return'''


def example_postprocess(response: str, dataset: str, loader):
    print(response)
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


qa = utils.load_qa(name="qa").select(range(10))
runner_lite = Runner(
    model_call=call_my_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, loader=load_sample
    ),
    qa=qa,
    batch_size=10,
)
evaluator = Evaluator(qa=qa)
responses_lite = runner_lite.run(save="predictions_lite_nosem.txt")
print(f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True) * 100} %") # ~30 %