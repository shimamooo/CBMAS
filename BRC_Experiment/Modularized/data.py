from typing import List, Tuple
import json

import pandas as pd
from datasets import load_dataset


def load_winogender_pairs(dataset_name: str = "oskarvanderwal/winogender", subset: str = "all") -> List[Tuple[str, str]]:
    """Load Winogender dataset and return list of (female_prompt, male_prompt) pairs.

    Pairs are grouped by occupation; we take the first example per (occupation, gender).
    """
    dataset = load_dataset(dataset_name, subset)
    df = dataset["test"].to_pandas()[["gender", "sentence", "occupation"]]
    df = df[df["gender"].isin(["female", "male"])]

    female_df = df[df["gender"] == "female"].groupby("occupation", as_index=False).first().set_index("occupation")
    male_df = df[df["gender"] == "male"].groupby("occupation", as_index=False).first().set_index("occupation")

    prompt_pairs: List[Tuple[str, str]] = []
    for occupation in female_df.index:
        if occupation in male_df.index:
            prompt_pairs.append((female_df.loc[occupation, "sentence"], male_df.loc[occupation, "sentence"]))
    return prompt_pairs


def load_reassurance_pairs(data_path: str = "data/reassurance.json") -> List[Tuple[str, str]]:
    """Load reassurance dataset and return list of (supportive_prompt, unsupportive_prompt) pairs.
    
    Each pair contains the question + choices + different answer selections:
    - supportive_prompt: question + choices + "\n\nI choose (1"  
    - unsupportive_prompt: question + choices + "\n\nI choose (2"
    
    This creates minimal pairs where the model predicts the choice number after
    "I choose (" which provides clearer choice differentiation and stronger signals.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    prompt_pairs: List[Tuple[str, str]] = []
    for item in data:
        question_with_choices = item["question"]
        supportive_prompt = f"{question_with_choices}\n\nI choose (1"
        unsupportive_prompt = f"{question_with_choices}\n\nI choose (2"
        prompt_pairs.append((supportive_prompt, unsupportive_prompt))
    
    return prompt_pairs


