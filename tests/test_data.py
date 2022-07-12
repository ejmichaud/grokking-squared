
from multigrok import data
import pytest

def test_mod_inverse():
    assert data.mod_inverse(2, 5) == 3
    assert data.mod_inverse(4, 5) == 4
    assert data.mod_inverse(1, 2) == 1
    assert data.mod_inverse(2, 3) == 2

def test_is_prime():
    assert data.is_prime(17)
    assert data.is_prime(101)
    assert data.is_prime(2)
    assert not data.is_prime(1)
    assert not data.is_prime(57)
    assert not data.is_prime(36)

def test_abelian_operators():
    assert all([(op in data.VALID_OPERATORS) for op in data.ABELIAN_OPERATORS])
    p = 17
    for op in data.ABELIAN_OPERATORS:
        op_fn = data.VALID_OPERATORS[op]
        for i in range(p):
            for j in range(i+1, p):
                assert op_fn(i, j, p) == op_fn(j, i, p)

def test_groups_codes():
    for operators in data.OPERATOR_GROUPS_CODES.values():
        assert all([(op in data.VALID_OPERATORS) for op in operators])

def test_ArithmeticDataset_0():
    dataset = data.ArithmeticDataset(['+'], p=2)
    assert len(dataset) == 4
    x, y = dataset[0]
    assert len(x) == 4
    assert dataset.equations.shape[0] == dataset.answers.numel()

def test_ArithmeticDataset_1():
    dataset = data.ArithmeticDataset(['+'], p=2, only_input_tokens=True)
    assert len(dataset) == 4
    x, y = dataset[0]
    assert len(x) == 2

def test_ArithmeticDataset_2():
    dataset = data.ArithmeticDataset(['/'], p=59)
    assert len(dataset) == 59**2 - 59 # make sure we never divide by zero

def test_ArithmeticDataset_3():
    dataset = data.ArithmeticDataset('BASIC2', p=59)
    assert len(dataset) == 2 * 59**2

def test_ArithmeticDataset_4():
    with pytest.raises(Exception):
        dataset = data.ArithmeticDataset('BASIC2', p=59, only_input_tokens=True)

def test_ArithmeticDataset_5():
    with pytest.raises(Exception):
        dataset = data.ArithmeticDataset('BASIC4', p=20)

def test_ArithmeticDataset_6():
    p = 17
    dataset = data.ArithmeticDataset('ALL', p=p)
    for i in range(len(dataset)):
        assert 0 <= dataset[i][0][0].item() < p
        assert 0 <= dataset[i][0][2].item() < p
        assert p <= dataset[i][0][1].item() < p + len(data.VALID_OPERATORS)
        assert dataset[i][0][3].item() == p + len(data.VALID_OPERATORS)

def test_ArithmeticDataset_7():
    p = 17
    dataset = data.ArithmeticDataset('ALL', p=p)
    for seq, answer in dataset:
        op = dataset.operation_from_token(seq[1].item())
        assert data.VALID_OPERATORS[op](seq[0].item(), seq[2].item(), p) == answer.item()
    

