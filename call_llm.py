#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from utils import load_json, save_json, load_prompt

# ========== Utils Function ==========

def clean_reasoning(reasoning):
    if not reasoning:
        return 'Proof has syntax error.'
    return reasoning

def clean_llm_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\n-", "\n", text)
    text = text.replace("**", "")

    return text.strip()

def parse_missing_logic(text):
    missing_SL = []
    missing_NL = []

    for line in text.split("\n"):
        if ":::" in line:
            sl, nl = line.split(":::", 1)
            missing_SL.append(sl.strip())
            missing_NL.append(nl.strip())

    return missing_SL, missing_NL

def format_new_sl_translation(result_item):
    temp = ''
    for one_SL in result_item['missing_SL']:
        temp += f"{one_SL} ::: NEW SL\n"
    fol = result_item['translation']['FOL']

    if '::: NEW SL' in fol:
        result = re.sub(r'^.*::: NEW SL\s*\n?', '', fol, flags=re.MULTILINE)
        fol = result.replace('\nPremises:\n',f'\nPremises:\n{temp}')
    else:
        fol = fol.replace('\nPremises:',f'\nPremises:\n{temp}')
        fol = fol.replace('\n\n', '\n')

    result_item['translation']['FOL'] = fol
    return result_item

def remove_solver_fields(item):
    fields = [
        "final_answer",
        "gold_answer",
        "solver_status",
        "solver_error",
        "reasoning"]

    for f in fields:
        item.pop(f, None)

def resume_from_output(output_file: str, data: List[Dict], results: List[Dict]):

    if not os.path.exists(output_file):
        return results, data

    valid_data = []
    existing_id = []
    new_result = []

    previous_results = load_json(output_file)

    for item in previous_results:
        current_id = item['id']
        if 'api_error' in item:
            continue
        existing_id.append(current_id)
        new_result.append(item)

    for item in data:
        current_id = item['id']
        if current_id not in existing_id:
            valid_data.append(item)

    print(f"Already finished: {len(new_result)}")
    print(f"Remaining tasks: {len(valid_data)}")

    return new_result, valid_data

def get_not_fact(input_file):
    data=load_json(input_file)
    invalid_data=[]
    valid_data=[]

    for item in data:
        if 'false' in item["llm_judge"].lower():
            invalid_data.append(item)       
        else:
            valid_data.append(item)

    save_json(input_file.replace('.json','_notfact.json'), invalid_data)
    save_json(input_file.replace('.json','_fact.json'), valid_data)

# ========== Call LLM ==========

class LLMHelper:
    def __init__(self, api_key: str, base_url: str, model: str):
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1 )
        return resp.choices[0].message.content


  
# ========== Task Option ==========

def translation(llm_helper, item):
    user_prompt=load_prompt('prompts/translation.txt')

    context = item.get('context', '')
    user_prompt = user_prompt.replace('${context}', context)
    
    question = item.get('question', '')
    user_prompt = user_prompt.replace('${question}', question)

    system_prompt = "You are an expert in symbolic logic translation. Follow the instructions carefully and provide accurate translations."
     
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    return llm_helper.chat(messages)


def logic_completion_init(llm_helper, item):

    user_prompt=load_prompt('prompts/logic_completion_init.txt')

    context = item.get('context', '')
    user_prompt = user_prompt.replace('${context}', context)
    
    question = item.get('question', '')
    user_prompt = user_prompt.replace('${conclusion}', question)

    reasoning = item.get('reasoning', '')
    user_prompt = user_prompt.replace('${reasoning}', reasoning)

    trans = item['translation']['FOL']
    user_prompt = user_prompt.replace('${translation}', trans)

    system_prompt = "You are an expert in symbolic logic reasoning. Follow the instructions carefully and provide accurate symbolic language."

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    return llm_helper.chat(messages)


def logic_completion(llm_helper, item):

    user_prompt=load_prompt('prompts/logic_completion.txt')
    
    context = item.get('context', '')
    user_prompt = user_prompt.replace('${context}', context)
    
    question = item.get('question', '')
    user_prompt = user_prompt.replace('${conclusion}', question)

    reasoning = item.get('prior_reasoning', '')
    if isinstance(reasoning, list):
        reasoning = reasoning[0] if reasoning else ''
    reasoning = clean_reasoning(reasoning)
    user_prompt = user_prompt.replace('${prior_reasoning}', reasoning)

    trans = item['translation']['FOL']
    if '::: NEW SL' in trans:
        trans = re.sub(r'^.*::: NEW SL\s*\n?', '', trans, flags=re.MULTILINE)
    user_prompt = user_prompt.replace('${translation}', trans)

    failed_formula = ''
    failed_reasoning = ''

    temp = ''
    for idx in range(len(item['missing_SL'])):
        temp += f"{item['missing_SL'][idx]} ::: {item['missing_NL'][idx]}\n"
    if 'prior_missing_logic' in item:
        for x in range(len(item['prior_missing_logic'])):
            failed_formula += ''.join(item['prior_missing_logic'][x])
            reasoning = item['prior_reasoning'][x+1]
            reasoning = clean_reasoning(reasoning)
            failed_reasoning += reasoning + '\n\n'
        failed_formula += temp
        reasoning = item.get('reasoning', '')
        reasoning = clean_reasoning(reasoning)
        failed_reasoning += reasoning

    else:
        failed_formula = temp
        reasoning = item.get('reasoning', '')
        reasoning = clean_reasoning(reasoning)
        failed_reasoning = reasoning

    user_prompt = user_prompt.replace('${failed_formula}', failed_formula)
    user_prompt = user_prompt.replace('${failed_reasoning}', failed_reasoning)

    system_prompt = "You are an expert in symbolic logic reasoning. Follow the instructions carefully and provide accurate symbolic language."

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    return llm_helper.chat(messages)


def fact_verification(llm_helper, item):

    user_prompt=load_prompt('prompts/fact_verification.txt')
    
    context=str(item["missing_NL"])
    user_prompt = user_prompt.replace('${premise}', context)

    system_prompt = "You are an AI specialized in evaluating the real-world truth of symbolic-logic premises. Follow the instructions carefully and provide accurate answer."
    
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    return llm_helper.chat(messages)


def fact_completion_init(llm_helper, item):

    user_prompt=load_prompt('prompts/fact_completion_init.txt')

    context = item.get('context', '')
    for one_premise in item['missing_NL']:
        context=context+'\n'+one_premise
    user_prompt = user_prompt.replace('${context}', context)
    
    question = item.get('question', '')
    user_prompt = user_prompt.replace('${conclusion}', question)

    reasoning = item.get('reasoning', '')
    user_prompt = user_prompt.replace('${reasoning}', reasoning)

    trans = item['translation']['FOL']
    user_prompt = user_prompt.replace('${translation}', trans)

    sentences=''
    llm_judge_list=item['llm_judge'].split('\n')
    false_indexes = [i for i, v in enumerate(llm_judge_list) if v=="False"]
    for idx in false_indexes:
        sentences=sentences+f"{item['missing_SL'][idx]} ::: {item['missing_NL'][idx]}\n"
    user_prompt = user_prompt.replace('${sentences}', sentences)

    system_prompt = "You are an expert in symbolic logic reasoning. Follow the instructions carefully and provide accurate information"
    
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    return llm_helper.chat(messages)


def fact_completion(llm_helper, item):

    user_prompt=load_prompt('prompts/fact_completion.txt')
    context = item.get('context', '').replace('\n\n','\n')
    for one_premise in item['missing_NL']:
        context=context+'\n'+one_premise
    user_prompt = user_prompt.replace('${context}', context)
    
    question = item.get('question', '')
    user_prompt = user_prompt.replace('${conclusion}', question)

    reasoning = item.get('reasoning', '')
    user_prompt = user_prompt.replace('${reasoning}', reasoning)

    trans = item['translation']['FOL']
    user_prompt = user_prompt.replace('${translation}', trans)

    sentences=''
    llm_judge_list=item['llm_judge'].split('\n')
    false_indexes = [i for i, v in enumerate(llm_judge_list) if v=="False"]
    for idx in false_indexes:
        sentences=sentences+f"{item['missing_SL'][idx]} ::: {item['missing_NL'][idx]}\n"
    user_prompt = user_prompt.replace('${sentences}', sentences)

    temp=''
    for ok in item['prior_missing_fact']:
        temp=temp+''.join(ok)
    user_prompt = user_prompt.replace('${missing_fact}', temp)

    system_prompt = "You are an expert in symbolic logic reasoning. Follow the instructions carefully and provide accurate information"

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    return llm_helper.chat(messages)

def call_cot(llm_helper, item):
    user_prompt=load_prompt('prompts/cot.txt')

    context = item.get('context', '')
    user_prompt = user_prompt.replace('${context}', context)
    
    question = item.get('question', '')
    user_prompt = user_prompt.replace('${question}', question)
    system_prompt = "You are an expert in symbolic logic reasoning. Follow the instructions carefully and provide accurate information"

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    

    return llm_helper.chat(messages)


def process_translation(result_item, cleaned_output):
    cleaned_output = cleaned_output.replace('  \n', '\n')
    cleaned_output = cleaned_output.replace('Conclusion\n', 'Conclusion:\n')
    cleaned_output = cleaned_output.replace('Predicates\n', 'Predicates:\n')
    cleaned_output = cleaned_output.replace('Premises\n', 'Premises:\n')

    if 'Premises:' not in cleaned_output:
        cleaned_output = '\nPremises:\n' + cleaned_output

    if 'Conclusion:' not in cleaned_output:
        lines = cleaned_output.strip().split('\n')
        cleaned_output = (
            '\n'.join(lines[:-1])
            + '\nConclusion:\n'
            + lines[-1]
        )

    result_item['translation'] = {"FOL": cleaned_output}

    return result_item

def process_logic_completion_init(result_item, cleaned_output):
    missing_SL, missing_NL = parse_missing_logic(cleaned_output)

    result_item['missing_SL'] = missing_SL
    result_item['missing_NL'] = missing_NL
    result_item['prior_reasoning'] = result_item['reasoning']
    result_item = format_new_sl_translation(result_item)
    remove_solver_fields(result_item)

    return result_item

def process_logic_completion(result_item, cleaned_output):
    missing_SL, missing_NL = parse_missing_logic(cleaned_output)
    temp = ""

    for idx in range(len(result_item['missing_SL'])):
        temp += (
            f"{result_item['missing_SL'][idx]} ::: "
            f"{result_item['missing_NL'][idx]}\n"
        )

    if 'prior_missing_logic' in result_item:
        result_item['prior_missing_logic'].append([temp])
    else:
        result_item['prior_missing_logic'] = [[temp]]

    if isinstance(result_item['prior_reasoning'], str):
        result_item['prior_reasoning'] = [result_item['prior_reasoning']]

    result_item['prior_reasoning'].append(result_item['reasoning'])
    result_item['missing_SL'] = missing_SL
    result_item['missing_NL'] = missing_NL
    result_item = format_new_sl_translation(result_item)
    remove_solver_fields(result_item)

    return result_item

def process_fact_verification(result_item, cleaned_output):

    result_item['llm_judge'] = cleaned_output

    return result_item

def process_fact_completion_init(result_item, cleaned_output):
    missing_SL, missing_NL = parse_missing_logic(cleaned_output)

    result_item['missing_SL'] = missing_SL
    result_item['missing_NL'] = missing_NL

    if 'solver_pass_num' not in result_item:
        result_item['solver_pass_num'] = 0

    temp = ""

    for idx in range(len(result_item['missing_SL'])):
        temp += (
            f"{result_item['missing_SL'][idx]} ::: "
            f"{result_item['missing_NL'][idx]}\n"
        )
    if 'prior_missing_fact' in result_item:
        result_item['prior_missing_fact'].append([temp])
    else:
        result_item['prior_missing_fact'] = [[temp]]
    result_item = format_new_sl_translation(result_item)
    return result_item

def process_fact_completion(result_item, cleaned_output):
    missing_SL, missing_NL = parse_missing_logic(cleaned_output)
    temp = ""

    for idx in range(len(result_item['missing_SL'])):
        temp += (
            f"{result_item['missing_SL'][idx]} ::: "
            f"{result_item['missing_NL'][idx]}\n"
        )

    result_item['prior_missing_fact'].append([temp])

    result_item['missing_SL'] = missing_SL
    result_item['missing_NL'] = missing_NL
    result_item = format_new_sl_translation(result_item)
    return result_item

def process_cot(result_item, cleaned_output):

    result_item['final_res'] = cleaned_output

    return result_item

TASK_FUNC = {
    "translation": translation,
    "logic_completion_init": logic_completion_init,
    "logic_completion": logic_completion,
    "fact_verification": fact_verification,
    "fact_completion_init": fact_completion_init,
    "fact_completion": fact_completion,
    "cot": call_cot
}

PROCESS_FUNC = {
    "translation": process_translation,
    "logic_completion_init": process_logic_completion_init,
    "logic_completion": process_logic_completion,
    "fact_verification": process_fact_verification,
    "fact_completion_init": process_fact_completion_init,
    "fact_completion": process_fact_completion,
    "cot": process_cot
}

# ========== Multi Worker Process ==========

def worker_process_item(item, api_key, base_url, model, task_type):
    llm_helper = LLMHelper(api_key=api_key, base_url=base_url, model=model)
    result_item = item.copy()

    try:
        task_func = TASK_FUNC[task_type]
        gpt_output = task_func(llm_helper, item)

        cleaned_output = clean_llm_output(gpt_output)

        processor = PROCESS_FUNC[task_type]
        result_item = processor(result_item, cleaned_output)

        return result_item
    
    except Exception as e:
        result_item["api_error"] = f"ExceptionError: {e}"
        return result_item


# ========== Main ==========

def main():
    
    parser = argparse.ArgumentParser(description='Call LLM')
    parser.add_argument('--input_file', type=str, default='', help='Input JSON file path')
    parser.add_argument('--output_file', type=str, default='', help='Output JSON file path')
    parser.add_argument('--api_key', type=str, default='', help='API key')
    parser.add_argument('--api_base_url', type=str, default='', help='API base URL')
    parser.add_argument('--model', type=str, default='', help='Model name')
    parser.add_argument('--num_workers', type=int, default=40, help='Num of workers')
    parser.add_argument('--task_type', type=str, choices=list(TASK_FUNC.keys()), default='', help='Task type')

    args = parser.parse_args()
    task_type=args.task_type
    data = load_json(args.input_file)

    max_retry=5
    results = []

    for _ in range(max_retry):

        results, data = resume_from_output(args.output_file, data, results)

        if len(data)==0:
            break

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(
                worker_process_item, 
                item, 
                args.api_key, 
                args.api_base_url, 
                args.model, 
                task_type) for item in data]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Calling LLM (multiprocess)"):
                result_item = fut.result()
                results.append(result_item)
                save_json(args.output_file, results)

        save_json(args.output_file, results)


    if task_type in ['fact_verification']:
        get_not_fact(args.output_file)


if __name__ == '__main__':
    main()