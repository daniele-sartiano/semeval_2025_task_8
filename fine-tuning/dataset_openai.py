import argparse
import json
import sys
import pandas as pd
import numpy as np

from databench_eval import Evaluator, utils

from openai import OpenAI

def patch_function(instruction: str):
    if instruction.startswith('return'):
        header = '''
def answer(df: pd.DataFrame):
    '''
        return f'{header}{instruction}'
    return instruction

def invoke_chatgpt(client, prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": "You are a pandas code generator."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def exec_function(dataset, exec_string):
    df = utils.load_table(dataset)
    local_vars = {"df": df , "pd": pd, "np": np}
    exec(exec_string + '\nans=answer(df)', local_vars)
    ans = local_vars['ans']

    if isinstance(ans, pd.Series):
        ans = ans.tolist()
    elif isinstance(ans, pd.DataFrame):
        ans = ans.iloc[:, 0].tolist()

    value = ans.split('\n')[0] if '\n' in str(ans) else ans

    return value

def main():

    parser = argparse.ArgumentParser(prog='')
    parser.add_argument('-model', default='gpt-4o', help='The openai model')
    parser.add_argument('-post-processing', action='store_true')
    parser.add_argument('-filter-truths', action='store_true')
    parser.add_argument('-split', default='train')
    
    args = parser.parse_args()

    if args.post_processing:
        for line in sys.stdin:
            row = json.loads(line.strip())
            out = {'prompt': row['prompt'], 'rejected': row['rejected'].replace('ans = answer(df)', '').strip(), 'chosen': row['chosen']}
            print(json.dumps(out))
    elif args.filter_truths:
        errors = 0
        qa = utils.load_qa(name="semeval", split=args.split, num_proc=32)
        evaluator = Evaluator(qa=qa)
        for i, line in enumerate(sys.stdin):
            entry = json.loads(line.strip())
            if qa[i]['question'] in entry['prompt']:
                row = qa[i]
            else:
                question = entry['prompt'].rsplit('Question: ')[-1].split('\n')[0]
                r = qa.filter(lambda x: question == x["question"])
                if r.shape[0] != 1:
                    print('error: find multiple results', question, file=sys.stderr)
                    errors += 1
                    continue
                row = r[0]
            try:
                result = exec_function(row['dataset'], entry['chosen'])
                
            except Exception as e:
                print(f'error: {e}', file=sys.stderr)
                continue
            
            if evaluator.compare(result, row['answer'], row['type']):
                print(line.strip())
            
            if i%50 == 0:
                print(f'done {i}', file=sys.stderr)
            
    else:
        client = OpenAI()

        for i, line in enumerate(sys.stdin):
            d = json.loads(line.strip())
            answer = invoke_chatgpt(client, d['prompt'], args.model)
            instruction = answer.replace('```python', '').replace('```', '').strip()
            output = {'prompt': d['prompt'], 
                    'rejected': d['exec_string'],
                    'instruction': instruction,
                    'chatgpt_raw_answer': answer,
                    'chosen': patch_function(instruction)
                    }
            print(json.dumps(output))
            if i % 50 == 0:
                print(f'Done {i}', file=sys.stderr)

if __name__ == '__main__':
    main()