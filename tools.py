
import argparse
import os, glob
import re, random
from tqdm import tqdm
import re
from call_llm import worker_process_item, resume_from_output
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import load_json, save_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_max_k(directory):
    max_num = -1

    for filename in os.listdir(directory):
        match = re.search(r'notfact_(\d+)', filename)
        if match:
            max_num = max(max_num, int(match.group(1)))

    return max_num

def extract_final_answer(final_res):
    patterns = [
        r'Conclusion\s*:?\s*\n?\s*([A-B])\)?',
        r'Conclusion\s*:?\s*.*?([A-B])\)?\s*$',
    ]

    for p in patterns:
        m = re.search(p, final_res, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper()

    matches = re.findall(r'\b([ABC])\)', final_res)
    if matches:
        return matches[-1].upper()

    matches = re.findall(r'\b([ABC])\b', final_res)
    if matches:
        return matches[-1].upper()

    return random.choice(['A', 'B'])

def merge_valid_data(folder_path, output_path, dataset):
    file_list=[]
    pattern = re.compile(rf"^{re.escape(dataset)}_\d+_valid\.json$", re.IGNORECASE)

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if pattern.fullmatch(file):
                file_list.append(os.path.join(root, file))
                
    new_data=[]
    valid_data_idx=[]
    for file in file_list:
        print(file, len(load_json(file)))
        for item in load_json(file):
            new_data.append(item)
            current_id=item['id']
            valid_data_idx.append(current_id)
    save_json(output_path, new_data)

    original_invalid_data=load_json(f'{folder_path}/{dataset}_invalid.json')
    new_invalid_data=[]
    for item in original_invalid_data:
        current_id=item['id']
        if current_id not in valid_data_idx:
            new_invalid_data.append(item)
    save_json(output_path.replace('_valid','_invalid'), new_invalid_data)

def get_final_data(input_file, output_file, dataset, api_key, api_base_url, model):
    final_data=[]
    max_k=get_max_k(f'{input_file}/sl2nl')

    sl2nl_file_list=glob.glob(f'{input_file}/sl2nl/*')
    solver_file_list=glob.glob(f'{input_file}/solver/*')

    A_file_list=[f'{input_file}/solver/{dataset}_valid.json']
    B_file_list=[]
    cot_file_list=[]

    for file in sl2nl_file_list:
        if file.endswith("_fact.json"):
            A_file_list.append(file)
        elif file.endswith("_notfact.json") and f'_notfact_{max_k}' in file:
            B_file_list.append(file)

    for file in solver_file_list:
        if file.endswith(f"_merge_invalid.json") :
            cot_file_list.append(file)
        elif file.endswith("_invalid.json") and '_notfact_' in file:
            cot_file_list.append(file)

    sample_a_list=[]
    sample_b_list=[]
    sample_cot_list=[]
    for file in A_file_list:
        sample_a_list=sample_a_list+load_json(file)
        print('A',file,len(load_json(file)))
    for file in B_file_list:
        sample_b_list=sample_b_list+load_json(file)
        print('B',file,len(load_json(file)))
    for file in cot_file_list:
        sample_cot_list=sample_cot_list+load_json(file)
        print('cot',file,len(load_json(file)))

    for item in sample_a_list:
        item['final_res']='A'
        final_data.append(item)
    for item in sample_b_list:
        item['final_res']='B'
        final_data.append(item)
    
    max_retry=5
    cot_results = []
    for _ in range(max_retry):
        cot_results, data = resume_from_output(output_file.replace(".json","_cot_tmp.json"), sample_cot_list, cot_results)

        if len(data)==0:
            break

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(worker_process_item, item, api_key, api_base_url, model, 'cot') for item in data]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Calling LLM (multiprocess)"):
                result_item = fut.result()
                cot_results.append(result_item)
                save_json(output_file.replace(".json","_cot_tmp.json"), cot_results)


    final_data=final_data+cot_results
    save_json(output_file, final_data)

def evaluation(input_file):
    data = load_json(input_file)

    y_true = []
    y_pred = []

    for item in data:
        gold = item["answer"]
        pred = item["final_res"]
        if pred!='A' and pred!='B':
            pred=extract_final_answer(pred)

        y_true.append(gold)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true,y_pred,average="binary",pos_label="A")
    recall = recall_score(y_true,y_pred,average="binary",pos_label="A")
    f1 = f1_score(y_true,y_pred,average="binary",pos_label="A")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description='call api')
    parser.add_argument('--input_file', type=str, default='', help='Input JSON file path')
    parser.add_argument('--output_file', type=str, default='', help='Output JSON file path')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name')
    parser.add_argument('--api_key', type=str, default='', help='API key')
    parser.add_argument('--api_base_url', type=str, default='', help='API base URL')
    parser.add_argument('--model', type=str, default='', help='Model name')
    parser.add_argument('--task_type', type=str, choices=['merge_valid_data','get_final_data','evaluation'], default='', help='Task type')


    args = parser.parse_args()
    task_type=args.task_type

    if task_type=='merge_valid_data':
        merge_valid_data(args.input_file, args.output_file, args.dataset)

    if task_type=='get_final_data':
        get_final_data(args.input_file, args.output_file, args.dataset, args.api_key, args.api_base_url, args.model)

    elif task_type=='evaluation':
        evaluation(args.input_file)

if __name__ == '__main__':
    main()

