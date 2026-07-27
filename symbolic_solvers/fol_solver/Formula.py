from nltk.tree import Tree
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fol_solver.fol_parser import FOL_Parser
import func_timeout

class FOL_Formula:
    def __init__(self, str_fol) -> None:
        self.parser = FOL_Parser()

        def parse_fol_with_timeout():
            """Parse FOL string with timeout handling"""
            return self.parser.parse_text_FOL_to_tree(str_fol)

        try:
            # Use func_timeout instead of signal.alarm to avoid global interruption
            tree = func_timeout.func_timeout(10, parse_fol_with_timeout)
        except func_timeout.FunctionTimedOut:
            # Handle timeout gracefully
            tree = None
            self._is_valid = False
            return
        except Exception as exc:
            # Handle other parsing errors
            tree = None
            self._is_valid = False
            return
    
        self.tree = tree
        if tree is None:
            self._is_valid = False
        else:
            self._is_valid = True
            self.variables, self.constants, self.predicates = self.parser.symbol_resolution(tree)
    
    def __str__(self) -> str:
        _, rule_str = self.parser.msplit(''.join(self.tree.leaves()))
        return rule_str
    
    @property
    def is_valid(self):
        return self._is_valid

    def _get_formula_template(self, tree, name_mapping):
        for i, subtree in enumerate(tree):
            if isinstance(subtree, str):
                # Modify the leaf node label
                if subtree in name_mapping:
                    new_label = name_mapping[subtree]
                    tree[i] = new_label
            else:
                # Recursive call to process this subtree
                self._get_formula_template(subtree, name_mapping)

    def get_formula_template(self):
        template = self.tree.copy(deep=True)
        name_mapping = {}
        for i, f in enumerate(self.predicates):
            name_mapping[f] = 'F%d' % i
        for i, f in enumerate(self.constants):
            name_mapping[f] = 'C%d' % i

        self._get_formula_template(template, name_mapping)
        self.template = template
        _, self.template_str = self.parser.msplit(''.join(self.template.leaves()))
        return name_mapping, self.template_str
        
    
if __name__ == '__main__':

    str_fol = '∀x (Top10(x) → ∃y (JapaneseCompany(y) ∧ CreatedBy(x,y)))'
    
    fol_rule = FOL_Formula(str_fol)
    if fol_rule.is_valid:
        print(fol_rule)
        name_mapping, template = fol_rule.get_formula_template()
