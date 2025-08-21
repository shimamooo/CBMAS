from typing import List, Tuple

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


