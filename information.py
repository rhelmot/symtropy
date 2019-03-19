import claripy
import random
from math import log2

# https://en.wikipedia.org/wiki/Entropy_(information_theory)

pcache = {}
def P(C, trials=3000):
    assert type(C) is claripy.ast.Bool

    o = pcache.get(C.cache_key, None)
    if o is not None:
        return o

    # quick answers
    qb = quick_bits(C)
    if qb is not None:
        return 1 / 2**len(qb)

    # hack to get around traversing the AST
    # YIKES FOREVER
    #roots = [claripy.BVS(n, int(n.split('_')[-1]), explicit_name=True) for n in set().union(*[c.variables for c in constraints])]
    roots = [claripy.BVS(n, int(n.split('_')[-1]), explicit_name=True) for n in C.variables]
    successes = 0
    for _ in range(trials):
        replacements = {root.cache_key: claripy.BVV(random.randrange(0, 2**len(root)), len(root)) if type(root) is claripy.ast.BV else claripy.true if random.randrange(0, 2) else claripy.false for root in roots}
        value = C.replace_dict(replacements)
        if value.is_true():
            successes += 1

    o = successes/trials
    pcache[C.cache_key] = o
    return o

def square_independent(A, B):
    pA = P(A)
    pB = P(B)
    pnA = 1-pA
    pnB = 1-pB
    return [[pnA*pnB, pnA*pB], [pA*pnB, pA*pB]]

def quick_bits(ast):
    if ast.op == 'And':
        try:
            return set().union(*(quick_bits(sub) for sub in ast.args))
        except TypeError:
            return None
    if ast.op == '__eq__':
        if ast.args[0].op == 'BVV':
            comp = ast.args[1]
        elif ast.args[1].op == 'BVV':
            comp = ast.args[0]
        else:
            return None
        return quick_bits(comp)
    if ast.op == 'Extract' and ast.args[2].op == 'BVS':
        return {(ast.args[2].args[0], i) for i in range(ast.args[1], ast.args[0]+1)}
    if ast.op == 'BVS':
        return {(ast.args[0], i) for i in range(len(ast))}
    return None

def square(A, B, trials=10000):
    # quick check for independence
    if not A.variables.intersection(B.variables):
        return square_independent(A, B)
    qa = quick_bits(A)
    qb = quick_bits(B)
    if qa is not None and qb is not None and not qa.intersection(qb):
        return square_independent(A, B)

    roots = [claripy.BVS(n, int(n.split('_')[-1]), explicit_name=True) for n in A.variables | B.variables]

    counts = [[0, 0], [0, 0]]
    for _ in range(trials):
        replacements = {root.cache_key: claripy.BVV(random.randrange(0, 2**len(root)), len(root)) if type(root) is claripy.ast.BV else claripy.true if random.randrange(0, 2) else claripy.false for root in roots}
        a_value = A.replace_dict(replacements).is_true()
        b_value = B.replace_dict(replacements).is_true()
        counts[a_value][b_value] += 1

    counts[True][True] /= trials
    counts[False][True] /= trials
    counts[True][False] /= trials
    counts[False][False] /= trials
    return counts

def H(C):
    try:
        return -(P(C)*log2(P(C)) + (1-P(C))*log2(1-P(C)))
    except ValueError: # there's a zero in a log
        return 0.

def CondH(C, Pred): # P(C|Pred)
    s = square(C, Pred)
    pp = s[True][True] + s[False][True] # P(Pred)
    ps = [1-pp, pp]

    result = 0
    for cv in (True, False):
        for pv in (True, False):
            try:
                result -= s[cv][pv]*log2(s[cv][pv]/ps[pv])
            except (ValueError, ZeroDivisionError):
                pass

    return result

def H1(Cn):
    result = 0.
    tail = None
    for C in Cn:
        if tail is None:
            result += H(C)
            tail = C
        else:
            result += CondH(C, tail)
            tail = claripy.And(C, tail)

    return result
