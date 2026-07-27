import os
import re,time
import sys
import traceback
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
import subprocess
import tempfile, textwrap, itertools as it

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.insert(0, project_root)

from symbolic_solvers.fol_solver.fol_prover9_parser import Prover9_FOL_Formula
from symbolic_solvers.fol_solver.Formula import FOL_Formula

# set the path to the prover9 executable
# PROVER9_PATH = os.path.join(os.path.dirname(__file__), '..', 'Prover9', 'bin')
# os.environ['PROVER9'] = PROVER9_PATH # Linux version
PROVER9_PATH = '/opt/homebrew/bin'
os.environ['PROVER9'] = PROVER9_PATH  # macOS version installed via Homebrew

TIMEOUT=2

# --- helper utilities for raw prover9 interaction starts (for result with unkown)---
def _build_p9_input(assumptions: list[str], goal: str, max_seconds) -> str:
    """Build a prover9 input string using NLTK's conversion utilities."""
    from nltk.inference.prover9 import Expression, convert_to_prover9

    ass_exprs = [Expression.fromstring(a) for a in assumptions]
    goal_expr = Expression.fromstring(goal)
    ass_strs = convert_to_prover9(ass_exprs)
    goal_str = convert_to_prover9(goal_expr)

    ass_block = "\n".join(a + "." for a in ass_strs)
    return textwrap.dedent(
        f"""
        assign(max_seconds,{max_seconds}).
        clear(auto_denials).

        formulas(assumptions).
        {ass_block}
        end_of_list.

        formulas(goals).
        {goal_str}.
        end_of_list.
    """
    )


def _run_prover9_raw(p9_input: str, timeout) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
        tf.write(p9_input)
        tf.flush()
        cmd = [os.path.join(PROVER9_PATH, "prover9"), "-f", tf.name]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    os.unlink(tf.name)
    return proc.stdout


_LINE_PAT = re.compile(r"^(Derived:|kept:|given\s+#\d+|-\w|\w).*?\[.*\]")


def _clean_prefix(line: str) -> str:
    """Strip technical prefixes so duplicates are detected."""
    line = re.sub(r"^(Derived:|kept:|given\s+#\d+)\s*", "", line.strip())
    line = re.sub(r"^\d+\s+", "", line)
    return line


def _summarise_log(log: str, max_lines: int | None = None) -> str:
    seen, text_seen, selected = set(), set(), []
    mapping: dict[int, int] = {}
    key_to_index: dict[str, int] = {}
    for ln in log.splitlines():
        if ln.startswith(("Predicate symbol precedence",
                          "Function symbol precedence",
                          "given #")):
            continue
        if _LINE_PAT.match(ln):
            m = re.match(r"\s*(\d+)\s+(.*)", ln.strip())
            if m:
                orig_no, raw_output = int(m.group(1)), m.group(2)
            else:
                orig_no, raw_output = None, re.sub(r"^\d+\s+", "", ln.strip())
            if raw_output.startswith("kept:"):
                raw_output = re.sub(r"^kept:\s*\d+\s*", "kept: ", raw_output)
            dedup_key = " ".join(_clean_prefix(raw_output).split())
            text_key = _clean_prefix(raw_output).split("[", 1)[0].strip()
            if raw_output.startswith("kept:") and text_key in text_seen:
                continue
            if dedup_key not in seen:
                selected.append((orig_no, raw_output))
                idx = len(selected)
                if orig_no is not None:
                    mapping[orig_no] = idx
                seen.add(dedup_key)
                key_to_index[dedup_key] = idx
                text_seen.add(text_key)
            else:
                # duplicate line, map its original number to existing index
                if orig_no is not None and dedup_key in key_to_index:
                    mapping[orig_no] = key_to_index[dedup_key]
    if max_lines:
        selected = selected[:max_lines]
        # rebuild mapping for truncated output
        mapping = {orig: idx + 1 for idx, (orig, _) in enumerate(selected) if orig is not None}
    out = []
    for idx, (orig_no, ln) in enumerate(selected, 1):
        clause, label = ln.rsplit("[", 1)
        clause_part = re.sub(r"^\d+\s+", "", clause)
        label = re.sub(r'(?<=\(|,)\d+(?=\)|,)', lambda m: str(mapping.get(int(m.group(0)), int(m.group(0)))), '[' + label)
        out.append(f"{idx} {clause_part.strip()} {label}")
    reason = "-- Search terminated, no contradiction found --" if "sos_empty" in log else \
             "-- Timeout terminated, no contradiction found --" if "max_seconds" in log else \
             "-- Search terminated, no contradiction found --"
    return "\n".join(out) + f"\n{reason}"

# --- helper utilities for raw prover9 interaction ends ---



class FOL_Prover9_Program:
    def __init__(self, logic_program:str, dataset_name = 'FOLIO') -> None:
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.flag = self.parse_logic_program()

    def parse_logic_program(self):

        modified_logic_program = re.sub(r'###\s*Predicates:', 'Predicates:', self.logic_program)
        modified_logic_program = re.sub(r'###\s*Premises:', 'Premises:', modified_logic_program)
        modified_logic_program = re.sub(r'###\s*Conclusion:', 'Conclusion:', modified_logic_program)
        modified_logic_program = re.sub(r'^\d+\.\s*', '', modified_logic_program, flags=re.MULTILINE)
        
        self.logic_program = modified_logic_program    
        # Original parsing logic for other datasets
        # Split the string into premises and conclusion
        premises_string = self.logic_program.split("Conclusion:")[0].split("Premises:")[1].strip()
        conclusion_string = self.logic_program.split("Conclusion:")[1].strip()

        # Extract each premise and the conclusion using regex
        premises = premises_string.strip().split('\n')
        conclusion = conclusion_string.strip().split('\n')

        self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
        self.logic_conclusion = conclusion[0].split(':::')[0].strip()
        
        max_retries=1
        try:
            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                if prover9_rule.formula==None:
                    while max_retries>0:     
                        prover9_rule = Prover9_FOL_Formula(fol_rule)
                        max_retries=max_retries-1
                        if prover9_rule.formula!=None:
                            break
                if prover9_rule.formula!=None:
                    self.prover9_premises.append(prover9_rule.formula)
            
            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            
            return True
        
        except Exception as e:
            print("Exception occurred:",e)
            traceback.print_exc()
            return False

    def execute_program(self):
        # Check if logic program parsing was successful
        if not self.flag:
            return None, "Logic program parsing failed", '', -1

        goal = Expression.fromstring(self.prover9_conclusion)
        assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
        timeout = TIMEOUT
        start_time = time.perf_counter()
        
        prover = Prover9Command(goal, assumptions, timeout=timeout)
        result = prover.prove()

        proof_trace = ''
        if result:
            proof_core = self._extract_proof_steps_ture_false(prover.proof(simplify=True))
            proof_trace = 'prove original conclusion:\n' + proof_core
            return 'True', '', proof_trace
        else:
            proof_trace += 'prove original conclusion:\n' + prover.proof(simplify=False) + '\n'

            negated_goal = NegatedExpression(goal)
            prover_neg = Prover9Command(negated_goal, assumptions, timeout=timeout)
            negation_result = prover_neg.prove()

            if negation_result:
                proof_core = self._extract_proof_steps_ture_false(prover_neg.proof(simplify=True))
                proof_trace = 'prove negation of original conclusion:\n' + proof_core
                return 'False', '', proof_trace
            else:
                orig_in  = _build_p9_input(self.prover9_premises, self.prover9_conclusion,max_seconds=timeout)
                orig_log = _run_prover9_raw(orig_in, timeout=timeout+0.5)
                orig_tr  = _summarise_log(orig_log)

                neg_goal = f"-({self.prover9_conclusion})"
                neg_in   = _build_p9_input(self.prover9_premises, neg_goal,max_seconds=timeout)
                neg_log  = _run_prover9_raw(neg_in, timeout=timeout+0.5)
                neg_tr   = _summarise_log(neg_log)

                proof_trace = (f"trying to prove original conclusion:\n{orig_tr}\n\n"
                                f"trying to prove negation of original conclusion:\n{neg_tr}\n\n"
                                f"So: Unknown")
                return 'Unknown', '', proof_trace
        
    def answer_mapping(self, answer):
        """
        Map the prover9 output to the appropriate dataset answer format.
        
        Args:
            answer: The prover9 output ('True', 'False', 'Unknown')
            
        Returns:
            str: The mapped answer for the specific dataset
        """
        
        if self.dataset_name == 'FOLIO':
            # FOLIO supports A/B/C (True/False/Unknown) - keep original logic
            if answer == 'True':
                return 'A'
            elif answer == 'False':
                return 'B'
            elif answer == 'Unknown':
                return 'C'
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset_name}')
        
        # Fallback for unrecognized answers
        raise ValueError(f'Answer "{answer}" not recognized for dataset "{self.dataset_name}"')
        
    @staticmethod
    def _extract_proof_steps_ture_false(proof_str: str) -> str:
        """Extract only the numbered step lines from a Prover9 proof output.

        Prover9 proof outputs often contain headers, footers, and comments in
        addition to the essential step lines that begin with an integer index.
        This helper keeps only lines that start with digits (optionally
        preceded by whitespace), which correspond to the step annotations we
        are interested in displaying.
        """
        step_lines = []
        for line in proof_str.splitlines():
            if re.match(r"^\s*\d+", line):
                step_lines.append(line)
        return "\n".join(step_lines)

if __name__ == "__main__":

    logic_program_s = """Predicates:\nGame(x) ::: x is a game\nJapaneseCompany(x) ::: x is a Japanese game company\nCreatedBy(x,y) ::: x is created by y\nTop10(x) ::: x is on the Top 10 list\nSellsOverMillion(x) ::: x sells more than one million copies\nPremises:\nCreatedBy(legendofzelda,japancompany) ::: A Japanese game company created the game the Legend of Zelda.\n∀x (Top10(x) → ∃y (JapaneseCompany(y) ∧ CreatedBy(x,y))) ::: All games on the Top 10 list are made by Japanese game companies.\n∀x (SellsOverMillion(x) → Top10(x)) ::: If a game sells more than one million copies, then it will be included in the Top 10 list.\nConclusion:\nTop10(legendofzelda) ::: The Legend of Zelda is on the Top 10 list."""
   
    prover9_program = FOL_Prover9_Program(logic_program_s)
    result, error_message, reasoning = prover9_program.execute_program()

    if reasoning:
        print('reasoning:', reasoning)
    else:
        print('error_message:', error_message)
    