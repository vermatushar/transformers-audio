from datasetprocessor import *
from datasets import load_dataset
import json
import pandas as pd

from datasets import Dataset

df = pd.DataFrame(heart_dataset)

dataset = Dataset.from_dict(heart_dataset)
example = dataset['train'][300]
print(example)
print(len(dataset))



