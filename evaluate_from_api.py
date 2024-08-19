import os
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse

# inser your api key here
API_KEY = ""


def get_client():
    if args.model_name in ["gpt-4", "gpt-4o"]:
        openai.api_key = API_KEY
        client = openai
    elif args.model_name in ["deepseek-chat", "deepseek-coder"]:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/")
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]:
        genai.configure(api_key=API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        client = genai.GenerativeModel(
            model_name=args.model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620"]:
        client = anthropic.Anthropic(
            api_key=API_KEY,
        )
    elif args.model_name in ["meta-llama/Meta-Llama-3.1-405B-Instruct"]:
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepinfra.com/v1/openai",
        )
    else:
        client = None
        print("For other model API calls, please implement the client definition method yourself.")
    return client


def call_api(client, instruction, inputs):
    start = time.time()
    if args.model_name in ["gpt-4", "gpt-4o", "deepseek-chat", "deepseek-coder",
                           "meta-llama/Meta-Llama-3.1-405B-Instruct"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=message_text,
            temperature=0,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]:
        chat_session = client.start_chat(
            history=[]
        )
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=4000,
            system="",
            messages=[
                {"role": "user", "content": instruction + inputs}
            ],
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    return result


def load_mmlu_pro_p():
    dataset_orig = load_dataset("TIGER-Lab/MMLU-Pro")
    dataset_modified = load_dataset("parquet", data_files={"test": "mmlupp.parquet"})
    # dataset_modified = load_dataset("parquet", data_files={"test": "0000.parquet"})
    test_df, val_df = dataset_modified["test"], dataset_orig["validation"]

    # Add logging for modification types
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
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        # Preserve modification information
        if 'is_modified' in each:
            each['is_modified'] = each['is_modified']
        if 'is_modified_non_llm' in each:
            each['is_modified_non_llm'] = each['is_modified_non_llm']
        if 'modification_type_non_llm' in each:
            each['modification_type_non_llm'] = each['modification_type_non_llm']
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJKL"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    pattern = r"answer is \(?([A-L])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-L])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-L]\b(?!.*\b[A-L]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    try:
        exist = True
        q_id = single_question["question_id"]
        for each in exist_result:
            if q_id == each["question_id"] and single_question["question"] == each["question"]:
                pred = extract_answer(each["model_outputs"])
                return pred, each["model_outputs"], exist, each.get('is_modified'), each.get(
                    'is_modified_non_llm'), each.get('modification_type_non_llm')
        exist = False
        category = single_question["category"]
        cot_examples = cot_examples_dict[category]
        question = single_question["question"]
        options = single_question["options"]
        prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
                 " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
            .format(category)
        for each in cot_examples:
            prompt += format_example(each["question"], each["options"], each["cot_content"])
        input_text = format_example(question, options)
        try:
            response = call_api(client, prompt, input_text)
            if not response:
                print(f"API returned empty response for question ID: {q_id}")
                print("Full data point:")
                print(json.dumps(single_question, indent=2))
                return None, None, exist, single_question.get('is_modified'), single_question.get(
                    'is_modified_non_llm'), single_question.get('modification_type_non_llm')
        except Exception as e:
            print(f"API call error for question ID {q_id}: {e}")
            print("Full data point:")
            print(json.dumps(single_question, indent=2))
            return None, None, exist, single_question.get('is_modified'), single_question.get(
                'is_modified_non_llm'), single_question.get('modification_type_non_llm')

        pred = extract_answer(response)
        if pred is None:
            print(f"Failed to extract answer for question ID: {q_id}")
            print("Full data point:")
            print(json.dumps(single_question, indent=2))
            print("API response:")
            print(response)
        return pred, response, exist, single_question.get('is_modified'), single_question.get(
            'is_modified_non_llm'), single_question.get('modification_type_non_llm')
    except Exception as e:
        print(f"Unexpected error in single_request for question ID {q_id}: {e}")
        print("Full data point:")
        print(json.dumps(single_question, indent=2))
        if 'response' in locals():
            print("API response:")
            print(response)
        return None, None, False, None, None, None

def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
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

                        is_correct = False
                        if not each["pred"]:
                            random.seed(12345)
                            x = random.randint(0, len(each["options"]) - 1)
                            is_correct = (x == each["answer_index"])
                        else:
                            is_correct = (each["pred"] == each["answer"])

                        # Update overall stats
                        if is_correct:
                            category_record[category]["overall"]["corr"] += 1
                        else:
                            category_record[category]["overall"]["wrong"] += 1

                        # Update LLM modified stats
                        if each.get("is_modified") == True:
                            if is_correct:
                                category_record[category]["llm_modified"]["corr"] += 1
                            else:
                                category_record[category]["llm_modified"]["wrong"] += 1

                        # Update non-LLM two wrong stats
                        if each.get("is_modified_non_llm") == True and each.get(
                                "modification_type_non_llm") == "two_wrong":
                            if is_correct:
                                category_record[category]["non_llm_two_wrong"]["corr"] += 1
                            else:
                                category_record[category]["non_llm_two_wrong"]["wrong"] += 1

                        # Update non-LLM correct and wrong stats
                        if each.get("is_modified_non_llm") == True and each.get(
                                "modification_type_non_llm") == "correct_and_wrong":
                            if is_correct:
                                category_record[category]["non_llm_correct_and_wrong"]["corr"] += 1
                            else:
                                category_record[category]["non_llm_correct_and_wrong"]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects):
    client = get_client()
    test_df, dev_df = load_mmlu_pro_p()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data):
            label = each["answer"]
            category = subject
            pred, response, exist, is_modified, is_modified_non_llm, modification_type_non_llm = single_request(client,
                                                                                                                each,
                                                                                                                dev_df,
                                                                                                                res)
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {
                        "overall": {"corr": 0.0, "wrong": 0.0},
                        "llm_modified": {"corr": 0.0, "wrong": 0.0},
                        "non_llm_two_wrong": {"corr": 0.0, "wrong": 0.0},
                        "non_llm_correct_and_wrong": {"corr": 0.0, "wrong": 0.0}
                    }
                each["pred"] = pred
                each["model_outputs"] = response
                each["is_modified"] = is_modified
                each["is_modified_non_llm"] = is_modified_non_llm
                each["modification_type_non_llm"] = modification_type_non_llm
                merge_result(res, each)

                is_correct = pred == label if pred is not None else False

                # Update overall stats
                category_record[category]["overall"]["corr" if is_correct else "wrong"] += 1

                # Update LLM modified stats
                if is_modified:
                    category_record[category]["llm_modified"]["corr" if is_correct else "wrong"] += 1

                # Update non-LLM stats
                if is_modified_non_llm:
                    if modification_type_non_llm == "two_wrong":
                        category_record[category]["non_llm_two_wrong"]["corr" if is_correct else "wrong"] += 1
                    elif modification_type_non_llm == "correct_and_wrong":
                        category_record[category]["non_llm_correct_and_wrong"]["corr" if is_correct else "wrong"] += 1

                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_summary(category_record, output_summary_path):
    for category, record in category_record.items():
        for key in record:
            corr = record[key]["corr"]
            wrong = record[key]["wrong"]
            total = corr + wrong
            if total > 0:
                record[key]["acc"] = corr / total
            else:
                record[key]["acc"] = 0.0

    with open(output_summary_path, "w") as fo:
        json.dump(category_record, fo, indent=2)

def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4o",
                        choices=["gpt-4", "gpt-4o", "deepseek-chat", "deepseek-coder",
                                 "gemini-1.5-flash-latest", "gemini-1.5-pro-latest",
                                 "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620",
                                 "meta-llama/Meta-Llama-3.1-405B-Instruct"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(assigned_subjects)