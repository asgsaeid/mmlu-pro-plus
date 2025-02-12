# MMLU-Pro+ Dataset and Evaluation 
🔵 **_NeurIPS 2024, Safe Generative AI_** 



This repository contains the MMLU-Pro+ dataset and evaluation scripts, an enhanced version of the MMLU-Pro benchmark designed to assess higher-order reasoning capabilities in Large Language Models (LLMs).

## Overview

MMLU-Pro+ introduces multi-choice questions with **multiple correct answers**, and distractors probing the higher-order reasoning capabilities and potential shortcut learning of LLMs. It is also available through EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/a40fe42aae2a5232448495c7bb5bf76aef100309/lm_eval/tasks/mmlu-pro-plus/README.md#L4).

## Models Tested

We evaluated the following state-of-the-art LLMs:

1. O1-preview
2. GPT-4o
3. Claude-Sonnet-3.5
4. Gemini-1.5-Pro
5. Llama-3.1-405B-Instruct
6. Qwen-2-72B-Instruct

## Evaluation Methods

We used API calls to evaluate the models:

- Gemini 1.5 Pro, GPT-4o, and O1-preview: We used their original APIs.
- Llama 3.1 405B Instruct and Qwen 2 72B Instruct: We used the API from [DeepInfra](https://deepinfra.com/).

## Evaluation Scripts

There are three main evaluation scripts:

1. `evaluate_from_api.py`: Can be used for all models.
2. `evaluate_from_api_multiprocess.py`: Same as above, but supports multi-processing if the API allows.
3. `evaluate_from_api_claude_rate_limit.py`: Specifically for Claude API, which has rate limits. You can set your own limits in the code.

## Results

Here are the accuracy results (%) on MMLU-Pro+ categories with performance drop from MMLU-Pro:

<img width="759" alt="image" src="https://github.com/user-attachments/assets/889b7705-3ddd-4fea-86eb-dedf492075c3">


## Additional Analyses

### Shortcut Learning Analysis

<img width="481" alt="image" src="https://github.com/user-attachments/assets/a730681f-c68c-4122-ba70-3db47daf724d">


### Correct Pair Identification (CPI) Analysis

<img width="622" alt="image" src="https://github.com/user-attachments/assets/93638947-974d-4a90-9d84-eb991ae626fc">



## Citation

If you use MMLU-Pro+ in your research, please cite our paper:

```bibtex
@article{taghanaki2024mmlu,
  title={MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs},
  author={Taghanaki, Saeid Asgari and Khani, Aliasgahr and Khasahmadi, Amir},
  journal={arXiv preprint arXiv:2409.02257},
  year={2024}
}
```


## Acknowledgements

We thank the creators of MMLU-Pro for their foundational work, which made MMLU-Pro+ possible.
