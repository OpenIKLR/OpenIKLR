
import argparse
from tqdm import tqdm
from typing import Dict, Tuple
from tools import load_json, save_json
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program

def get_not_approved(input_path):
    data=load_json(input_path)
    invalid_data=[]
    valid_data=[]
    for item in data:
        if item["solver_status"] == "success" and item["final_answer"] == "A":
            valid_data.append(item)
        else:
            invalid_data.append(item)

    save_json(input_path.replace('.json','_invalid.json'),invalid_data)
    save_json(input_path.replace('.json','_valid.json'),valid_data)


def get_gold_answer(item: Dict) -> str:
    answer = item.get('answer', '')
    if answer in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return answer
    elif answer in ['True', 'False']:
        return 'A' if answer == 'True' else 'B'
    else:
        if isinstance(answer, str) and len(answer) > 0:
            if len(answer) >= 2 and answer[1] == ')':
                return answer[0].upper()
        return answer



def execute_solver(sl: str, translation: str, item: Dict, dataset_name="FOLIO") -> Tuple[str, str, str, str]:

    try:
        dataset_name = dataset_name
        solver_class = FOL_Prover9_Program
        program = solver_class(translation, dataset_name)

        if not getattr(program, 'flag', True):
            return 'A', 'parsing error', 'Failed to parse symbolic program', ''
        
        try:
            answer, err, reasoning = program.execute_program()
        except Exception as e:
            return 'A', 'execution error', str(e), ''
        
        if answer is None:
            err_str = str(err) if err is not None else 'Unknown error'
            return 'A', 'execution error', err_str, ''
        
        mapped = program.answer_mapping(answer)
        
        status_code = 'success'
        error_message = ''
        if reasoning == '':
            status_code = 'execution error'
            error_message = 'Empty reasoning indicates execution failure'
        
        return mapped, status_code, error_message, reasoning
        
    except Exception as e:
        return 'A', 'execution error', str(e), ''



def call_solver(data, output_file_pat):
    results = []
    for item in tqdm(data, desc="Solving problems"):
        sl = "FOL"
        translation = item['translation'][sl]
        answer, status_code, error_message, reasoning = execute_solver(sl, translation, item)
        result_item = item.copy()
        result_item['final_answer'] = answer
        result_item['gold_answer'] = get_gold_answer(item)
        result_item['solver_status'] = status_code
        result_item['solver_error'] = error_message
        result_item['reasoning'] = reasoning
        
        if 'solver_pass_num' in result_item:
            if answer=='A':
                result_item['solver_pass_num']+=1
        
        results.append(result_item)
    
    save_json(output_file_pat, results)
    print(f"\nResults saved to {output_file_pat}")

    return output_file_pat


def main():
    parser = argparse.ArgumentParser(description='call solver')
    parser.add_argument('--input_file', type=str, default='', help='Input JSON file path')
    parser.add_argument('--output_file', type=str, default='', help='Output JSON file path')

    args = parser.parse_args()
    data=load_json(args.input_file)
    call_solver(data, args.output_file)

    get_not_approved(args.output_file)
    

if __name__ == '__main__':
    main()
    