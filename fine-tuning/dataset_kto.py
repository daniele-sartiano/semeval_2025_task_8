import argparse
import json
import sys
import pandas as pd
import numpy as np

from databench_eval import Evaluator, utils

from openai import OpenAI

from transformers import set_seed

def main():
    evaluator = Evaluator()
    for line in sys.stdin:
        entry = json.loads(line.strip())
        if evaluator.compare(entry['answer'], entry['truth']['answer'], entry['truth']['type']):
            print(json.dumps({'prompt': entry['prompt'], 'completion': entry['answer'], 'label': False}))

if __name__ == '__main__':
    main()