
from collections import defaultdict
import torch

def mod_inverse(b, p):
    return (b ** (p-2)) % p

def is_prime(n):
    if n == 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

VALID_OPERATORS = {
    "+": lambda x, y, p: (x + y) % p,
    "-": lambda x, y, p: (x - y) % p,
    "*": lambda x, y, p: (x * y) % p,
    "/": lambda x, y, p: (x * mod_inverse(y, p)) % p,
    "**2+": lambda x, y, p: (x**2 + y) % p,
    "**3+": lambda x, y, p: (x**3 + y) % p,
    "x**2+y**2_mod_p": lambda x, y, p: (x**2 + y**2) % p,
    "x**2+y**2+x*y_mod_p": lambda x, y, p: (x**2 + y**2 + x*y) % p,
    "x**2+y**2+x*y+x_mod_p": lambda x, y, p: (x**2 + y**2 + x*y + x) % p,
    "x**3+x*y_mod_p": lambda x, y, p: (x**3 + x*y) % p,
    "x**3+x*y**2+y_mod_p": lambda x, y, p: (x**3 + x*(y**2) + y) % p
}

ABELIAN_OPERATORS = [
    "+", 
    "*", 
    "x**2+y**2_mod_p", 
    "x**2+y**2+x*y_mod_p"
]

# provide some shorthands for subsets of VALID_OPERATIONS
OPERATOR_GROUPS_CODES = {
    "BASIC2": ["+", "*"],
    "BASIC4": ["+", "*", "-", "/"],
    "BASIC6": ["+", "*", "-", "/", "**2+", "**3+"],
    "ABELIANS": ABELIAN_OPERATORS,
    "ALL": list(VALID_OPERATORS)
}


class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, operators, p=59, halve_abelian=False, only_input_tokens=False):
        """A dataset of arithmetic equations for studying grokking.

        Args:
            operators (list or str): specify the operations in the dataset with a list
                of str e.g. ["+", "-", "**2+"] or a special code: 
                "BASIC2", "BASIC4", "BASIC6", "ABELIANS", "ALL".
            p (int): modulus.
            halve_abelian (bool): whether to exclude half the equations if the operation is abelian.
            only_input_tokens (bool): if true, equations are just 2-token <a><b>, and don't include a token
                for the operation. Errors if there is more than one operator in `operators`. 
        """
        if type(operators) is list:
            assert len(operators) >= 1, "Must give at least one operator in `operators`"
            assert all([x in VALID_OPERATORS for x in operators]), "Invalid operation included in `operators`"
            self.operators = operators
        elif type(operators) is str and operators in OPERATOR_GROUPS_CODES:
            self.operators = OPERATOR_GROUPS_CODES[operators]
        else:
            raise Exception(f"`operators`: {operators} ({type(operators)}) not valid.")
        
        # Tokens are ordered like so:
        # 0                   -> 0
        # 1                   -> 1
        # ...                 ...
        # p-1                 -> p-1
        # p                   -> op1
        # p + 1               -> op2
        # ...                 ...
        # p + |operators| - 1 -> op_|operators|
        # p + |operators|     -> =
        # p + |operators| + 1 -> ?
        if "/" in self.operators and not is_prime(p):
            raise Exception("Modulus must be prime if division '/' is in `operations`.")
        self.p = p
        self.equals_token = p + len(self.operators)
        self.blank_token = p + len(self.operators) + 1
        self._ntokens = p + len(self.operators) + 2 # the + 2 is for '=' and '?'

        if only_input_tokens and len(self.operators) > 1:
            raise Exception("Must include token for operation when there is more than one operation.")
        
        # create data, organized by operation
        equations, answers = [], []
        for k, op in enumerate(self.operators):
            if halve_abelian and op in ABELIAN_OPERATORS:
                for x in range(p):
                    for y in range(x, p):
                        if not (op == '/' and y == 0):
                            if only_input_tokens:
                                equations.append([x, y]) # <x><y>
                            else:
                                equations.append([x, p+k, y, self.equals_token]) # <x><op><y><=>
                            answers.append([VALID_OPERATORS[op](x, y, p)])
            else:
                for x in range(p):
                    for y in range(p):
                        if not (op == '/' and y == 0):
                            if only_input_tokens:
                                equations.append([x, y]) # <x><y>
                            else:    
                                equations.append([x, p+k, y, self.equals_token])
                            answers.append([VALID_OPERATORS[op](x, y, p)])
        self.equations = torch.tensor(equations)
        self.answers = torch.tensor(answers).flatten()
    
    def __len__(self):
        return self.equations.shape[0]
    
    def __getitem__(self, idx):
        return self.equations[idx], self.answers[idx]
    
    @property
    def ntokens(self):
        return self._ntokens
    
    @property
    def sequence_length(self):
        return self.equations.shape[1]

    def operation_from_token(self, tok):
        """Returns operation (str) given token (int).
        """
        if type(tok) is torch.Tensor:
            tok = tok.item()
        return self.operators[tok - self.p]

    def readable_equation(self, sequence) -> str:
        symbols = []
        for i in sequence:
            if i < self.p:
                symbols.append(str(i.item()))
            elif i >= self.p and i < self.equals_token:
                symbols.append(self.operators[i - self.p])
            elif i == self.equals_token:
                symbols.append('=')
            else:
                raise Exception(f"Unrecognized token: {i}")

        return "".join(symbols)



