from typing import List, Tuple, Optional
import json
import os
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




def load_train_dataset(dataset_name: str, data_base_path: str = "data") -> List[Tuple[str, str]]:
    """Load training dataset and return list of (positive_prompt, negative_prompt) pairs.

    This function loads train data from the specified dataset directory and creates
    positive/negative prompt pairs with choices pre-filled for building steering vectors.

    Args:
        dataset_name: Name of the dataset (e.g., "deference", "reassurance", "satisficing", "sycophancy")
        data_base_path: Base path where dataset directories are located

    Returns:
        List of tuples with (positive_prompt, negative_prompt) pairs for training
    """
    data_path = os.path.join(data_base_path, dataset_name, f"{dataset_name}_train.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    prompt_pairs: List[Tuple[str, str]] = []

    for item in data:
        question_with_choices = item["question"]

        # Create prompts with the appropriate choice pre-filled
        matching_choice = item["answer_matching_behavior"]
        non_matching_choice = item["answer_not_matching_behavior"]

        # Extract the choice number/letter "(A)" -> "A")
        matching_choice_clean = matching_choice.strip("()")
        non_matching_choice_clean = non_matching_choice.strip("()")

        # Create prompts that end just before the choice
        base_prompt = f"{question_with_choices}\n\nI choose ("

        # Add the specific choice for each prompt
        positive_prompt = f"{base_prompt}{matching_choice_clean}"
        negative_prompt = f"{base_prompt}{non_matching_choice_clean}"

        prompt_pairs.append((positive_prompt, negative_prompt))

    return prompt_pairs


def load_test_dataset(dataset_name: str, data_base_path: str = "data") -> List[str]:
    """Load test dataset and return list of base prompts without choices filled in.

    This function loads test data from the specified dataset directory and returns
    base prompts for evaluation (letting the model generate choices naturally).

    Returns:
        List of base prompts for testing (without choices pre-filled)
    """
    data_path = os.path.join(data_base_path, dataset_name, f"{dataset_name}_test.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test dataset file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    base_prompts: List[str] = []

    for item in data:
        question_with_choices = item["question"]

        # For testing, we want the model to generate the choice naturally
        # So we use the base prompt that ends with "I choose ("
        base_prompt = f"{question_with_choices}\n\nI choose ("
        base_prompts.append(base_prompt)

    return base_prompts
