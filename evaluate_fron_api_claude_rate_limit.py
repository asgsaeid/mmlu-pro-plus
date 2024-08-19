import os
import anthropic
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse

# inser your api key here
API_KEY = ""

# Rate limiting constants based on your account
REQUESTS_PER_MINUTE = 2000
TOKENS_PER_MINUTE = 160000
TOKENS_PER_DAY = 5000000

# Global variables for rate limiting
requests_this_minute = 0
tokens_this_minute = 0
tokens_today = 0
last_reset_time = time.time()
day_start_time = time.time()


def get_client():
    return anthropic.Anthropic(api_key=API_KEY)


def reset_counters():
    global requests_this_minute, tokens_this_minute, tokens_today, last_reset_time, day_start_time
    current_time = time.time()

    if current_time - day_start_time >= 86400:
        tokens_today = 0
        day_start_time = current_time

    if current_time - last_reset_time >= 60:
        requests_this_minute = 0
        tokens_this_minute = 0
        last_reset_time = current_time


def wait_if_needed(tokens):
    global requests_this_minute, tokens_this_minute, tokens_today

    while True:
        reset_counters()

        if (requests_this_minute < REQUESTS_PER_MINUTE and
                tokens_this_minute + tokens <= TOKENS_PER_MINUTE and
                tokens_today + tokens <= TOKENS_PER_DAY):
            requests_this_minute += 1
            tokens_this_minute += tokens
            tokens_today += tokens
            return

        time.sleep(1)


def call_api_with_rate_limit(client, instruction, inputs, max_retries=5, initial_wait=1, max_wait=60):
    estimated_tokens = len(instruction.split()) + len(inputs.split())

    for attempt in range(max_retries):
        try:
            wait_if_needed(estimated_tokens)

            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": instruction + inputs}
                ],
                temperature=0.0,
            )
            response = message.content[0].text

            response_tokens = len(response.split())
            wait_if_needed(response_tokens)

            return response

        except anthropic.InternalServerError as e:
            if "overloaded_error" in str(e):
                wait_time = min(initial_wait * (2 ** attempt) + random.uniform(0, 1), max_wait)
                print(f"API overloaded. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

    print(f"API still overloaded after {max_retries} attempts. Skipping this request.")
    return None


def load_mmlu_pro_p():
    dataset_orig = load_dataset("TIGER-Lab/MMLU-Pro")
    dataset_modified = load_dataset("parquet", data_files={"test": "mmlupp.parquet"})
    # dataset_modified = load_dataset("parquet", data_files={"test": "0000.parquet"})
    test_df, val_df = dataset_modified["test"], dataset_orig["validation"]

    modification_counts = {
        "two_wrong": 0,
        "correct_and_wrong": 0,
        "llm_modified": 0,
        "none": 0
    }

    for item in test_df:
        if item.get('is_modified') == True:
            modification_counts["llm_modified"] += 1
        elif item.get('is_modified_non_llm') == True:
            mod_type = item.get('modification_type_non_llm')
            if mod_type in modification_counts:
                modification_counts[mod_type] += 1
            else:
                print(f"Unexpected modification type: {mod_type}")
        else:
            modification_counts["none"] += 1

    print("Modification type counts:")
    print(json.dumps(modification_counts, indent=2))

    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res = {}
    for each in test_df:
        options = [opt for opt in each["options"] if opt != "N/A"]
        each["options"] = options
        category = each["category"]
        if category not in res:
            res[category] = []
        res[category].append(each)
    return res


def format_example(question, options, cot_content=""):
    if not cot_content:
        cot_content = "Let's think step by step."
    example = f"Question: {question}\nOptions: "
    example += "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
    example += f"\nAnswer: {cot_content}\n\n"
    return example


def extract_answer(text):
    if text is None:
        return None

    patterns = [
        r"answer is \(?([A-L])\)?",
        r'.*[aA]nswer:\s*([A-L])',
        r"\b([A-L])\b(?!.*\b[A-L]\b)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)

    return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], True, each.get('is_modified'), each.get(
                'is_modified_non_llm'), each.get('modification_type_non_llm')

    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]

    prompt = f"The following are multiple choice questions (with answers) about {category}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    prompt += "".join([format_example(ex["question"], ex["options"], ex["cot_content"]) for ex in cot_examples])
    prompt += format_example(question, options)

    response = call_api_with_rate_limit(client, prompt, "")

    if response is None:
        print("API call returned None")
        return None, None, False, single_question.get('is_modified'), single_question.get(
            'is_modified_non_llm'), single_question.get('modification_type_non_llm')

    pred = extract_answer(response)
    return pred, response, False, single_question.get('is_modified'), single_question.get(
        'is_modified_non_llm'), single_question.get('modification_type_non_llm')


def update_result(output_res_path):
    category_record = {}
    res = []
    if os.path.exists(output_res_path):
        with open(output_res_path, "r") as fi:
            res = json.load(fi)
            for each in res:
                category = each["category"]
                if category not in category_record:
                    category_record[category] = {
                        "overall": {"corr": 0.0, "wrong": 0.0},
                        "llm_modified": {"corr": 0.0, "wrong": 0.0},
                        "non_llm_two_wrong": {"corr": 0.0, "wrong": 0.0},
                        "non_llm_correct_and_wrong": {"corr": 0.0, "wrong": 0.0}
                    }

                is_correct = each["pred"] == each["answer"] if each["pred"] else False

                category_record[category]["overall"]["corr" if is_correct else "wrong"] += 1

                if each.get("is_modified") == True:
                    category_record[category]["llm_modified"]["corr" if is_correct else "wrong"] += 1

                if each.get("is_modified_non_llm") == True:
                    if each.get("modification_type_non_llm") == "two_wrong":
                        category_record[category]["non_llm_two_wrong"]["corr" if is_correct else "wrong"] += 1
                    elif each.get("modification_type_non_llm") == "correct_and_wrong":
                        category_record[category]["non_llm_correct_and_wrong"]["corr" if is_correct else "wrong"] += 1

    return res, category_record


def save_summary(category_record, output_summary_path):
    for category, record in category_record.items():
        for key in record:
            corr = record[key]["corr"]
            wrong = record[key]["wrong"]
            total = corr + wrong
            record[key]["acc"] = corr / total if total > 0 else 0.0

    with open(output_summary_path, "w") as fo:
        json.dump(category_record, fo, indent=2)


def save_res(res, output_res_path):
    unique_res = {each["question_id"]: each for each in res}.values()
    with open(output_res_path, "w") as fo:
        json.dump(list(unique_res), fo)


import os
import anthropic
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse

# ... [previous code remains unchanged] ...

def evaluate(category=None):
    client = get_client()
    test_df, dev_df = load_mmlu_pro_p()

    categories_to_evaluate = [category] if category else test_df.keys()

    for current_category in categories_to_evaluate:
        if current_category not in test_df:
            print(f"Error: Category '{current_category}' not found in the dataset.")
            continue

        test_data = test_df[current_category]
        output_res_path = os.path.join(args.output_dir, f"{current_category}_result.json")
        output_summary_path = os.path.join(args.output_dir, f"{current_category}_summary.json")
        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data, desc=f"Processing {current_category}"):
            label = each["answer"]

            pred, response, exist, is_modified, is_modified_non_llm, modification_type_non_llm = single_request(client,
                                                                                                                each,
                                                                                                                dev_df,
                                                                                                                res)

            if response is not None:
                each["pred"] = pred
                each["model_outputs"] = response
                each["is_modified"] = is_modified
                each["is_modified_non_llm"] = is_modified_non_llm
                each["modification_type_non_llm"] = modification_type_non_llm
                res.append(each)

                is_correct = pred == label if pred is not None else False

                if current_category not in category_record:
                    category_record[current_category] = {
                        "overall": {"corr": 0.0, "wrong": 0.0},
                        "llm_modified": {"corr": 0.0, "wrong": 0.0},
                        "non_llm_two_wrong": {"corr": 0.0, "wrong": 0.0},
                        "non_llm_correct_and_wrong": {"corr": 0.0, "wrong": 0.0}
                    }

                category_record[current_category]["overall"]["corr" if is_correct else "wrong"] += 1

                if is_modified:
                    category_record[current_category]["llm_modified"]["corr" if is_correct else "wrong"] += 1

                if is_modified_non_llm:
                    if modification_type_non_llm == "two_wrong":
                        category_record[current_category]["non_llm_two_wrong"]["corr" if is_correct else "wrong"] += 1
                    elif modification_type_non_llm == "correct_and_wrong":
                        category_record[current_category]["non_llm_correct_and_wrong"]["corr" if is_correct else "wrong"] += 1

                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--category", "-c", type=str, help="Specify the category to evaluate (optional)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args.category)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--category", "-c", type=str, required=True, help="Specify the category to evaluate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args.category)