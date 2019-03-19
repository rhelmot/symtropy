import claripy
from functools import reduce
from collections import defaultdict, Counter
import information
import logging

l = logging.getLogger('symtropy.interpret')

cache = {} # translation cache

def interpret(ast):
    """
    Find the bits that are present in the ast. Will return a list where each entry corresponds
    to the corresponding bit of the AST. Entries can be booleans, indicating a concrete value,
    or a set of bits. The bits will be tuples of (AST, bit index).
    itself.

    Note: pretty much everything but the input ast will actually be a cache key. you have been
    warned.
    """
    if not isinstance(ast, claripy.ast.Base):
        import ipdb; ipdb.set_trace()

    if ast.cache_key in cache:
        return cache[ast.cache_key]
    else:
        raw_val = _interpret(ast)
        #val = raw_val
        val = manage_mix_masks(ast, raw_val)

        cache[ast.cache_key] = val
        return val

def _interpret(ast):
    if ast.op == 'BVV':
        return [bool((ast.args[0] >> i) & 1) for i in range(len(ast))]
    if ast.op == 'BVS':
        return [{(ast.cache_key, i)} for i in range(len(ast))]
    if ast.op == 'BoolV':
        return [ast.args[0]]
    if ast.op == 'BoolS':
        return [{(ast.cache_key, 0)}]
    if ast.op == '__neg__':
        return interpret(~ast.args[0] + 1)
    if ast.op == '__sub__':
        # generalize add-of-negate by simplifying a + (~b + 1) + (~c + 1) to a + ~b + ~c + 2
        return interpret(claripy.ast.BV('__add__', [ast.args[0]] + [~arg for arg in ast.args[1:]] + [claripy.BVV(len(ast.args - 1), len(ast))]))
    if ast.op == 'Extract':
        return [val for i, val in enumerate(interpret(ast.args[2])) if ast.args[1] <= i <= ast.args[0]]

    children = [interpret(child) for child in ast.args]

    if type(ast) == claripy.ast.bool.Bool:
        return [mix_all_bits(children)]
    if ast.op == '__invert__':
        return [~val if type(val) is bool else val for val in children]
    if ast.op in ('__and__', '__or__', '__xor__'):
        out = children[0]
        midout = []
        for child in children[1:]:
            midout = out
            out = []
            for bitA, bitB in zip(child, midout):
                if type(bitA) is bool and type(bitB) is bool:
                    if ast.op == '__and__':
                        out.append(bitA & bitB)
                    elif ast.op == '__or__':
                        out.append(bitA | bitB)
                    else:
                        out.append(bitA ^ bitB)
                elif (bitA == 0 or bitB == 0) and ast.op == '__and__':
                    out.append(0)
                elif (bitA == 1 or bitB == 1) and ast.op == '__or__':
                    out.append(1)
                else:
                    out.append(mix_bits([bitA, bitB]))

        return out

    if ast.op == '__add__':
        out = children[0]
        midout = []
        carry = False
        for child in children[1:]:
            midout = out
            out = []
            for bitA, bitB in zip(child, midout):
                if type(bitA) is bool and type(bitB) is bool and type(carry) is bool:
                    out.append(bitA ^ bitB ^ carry)
                    carry = (bitA & bitB) | (out[-1] & carry)
                else:
                    out.append(mix_bits([bitA, bitB, carry]))
                    carry = out[-1]

        return out

    if ast.op == '__mul__':
        sbit = 0
        int_children = [0]*len(children)
        for sbit in range(len(ast)):
            if any(type(child[sbit]) is not bool for child in children):
                break
            for i, child in enumerate(children):
                int_children[i] <<= 1
                int_children[i] += int(child[sbit])
        else:
            sbit = len(ast)

        int_result = reduce(lambda x, y: x * y, int_children)

        out = []
        carry = False
        for bit in range(len(ast)):
            if bit < sbit:
                out.append(bool((int_result >> bit) & 1))
            else:
                out.append(mix_bits([carry] + [child[bit] for child in children]))
                carry = out[-1]

        return out

    if ast.op in ('__div__', '__floordiv__', '__truediv__', '__mod__', 'SDiv', 'SMod'):
        top_bit = len(ast) - 1
        while top_bit >= 0 and children[1][top_bit] == 0:
            top_bit -= 1

        if top_bit == -1:
            raise ZeroDivisionError

        # div: you lose this many bits from the bottom
        # mod: you keep this many bits from the bottom
        # we want to capture that there's a partial bit of information when the divisor isn't 2^i
        # but, uh, how
        # div case is actually even more complicated bc it might be losing less if top bits are controlled
        # this whole piece is a whole series of approximations more or less equivalent to putting my fingers in my ears and screaming

        if ast.op in ('__mod__', 'SMod'):
            bits = children[0][:top_bit]
            have_bits = top_bit
        else:
            bits = children[0][top_bit:]
            have_bits = len(ast) - top_bit + 1

        if ast.op in ('SDiv', 'SMod'):
            bits += [children[0][-1]]

        if (ast.args[1] != 1 << top_bit).is_true():
            top_bit += 1
            have_bits += 1
            bits += [{((ast.args[0] % (1 << top_bit) > ast.args[1]).cache_key, 0)}]

        bit = mix_bits(bits)

        if ast.op == 'SDiv':
            return [bit]*len(ast)
        else:
            return [bit]*have_bits + [False]*(len(ast) - have_bits)

    if ast.op == 'If':
        out = []
        bitC = children[0][0]
        for bitA, bitB in zip(children[1], children[2]):
            if type(bitC) is bool:
                out.append(bitA if bitC else bitB)
            elif type(bitA) is bool and type(bitB) is bool and bitA == bitB:
                out.append(bitA)
            else:
                out.append(mix_bits([bitA, bitB, bitC]))

        return out

    if ast.op == '__rshift__':
        if ast.args[1].symbolic:
            print('what?')
        else:
            return [children[0][i + ast.args[1].args[0] if i + ast.args[1].args[0] < len(ast.args[0]) else len(ast.args[0]) - 1] for i in range(len(ast))]
    if ast.op == '__lshift__':
        if ast.args[1].symbolic:
            print('what?')
        else:
            return [children[0][i - ast.args[1].args[0]] if i - ast.args[1].args[0] >= 0 else 0 for i in range(len(ast))]
    if ast.op == 'LShR':
        if ast.args[1].symbolic:
            print('what?')
        else:
            return [children[0][i + ast.args[1].args[0]] if i + ast.args[1].args[0] < len(ast.args[0]) else 0 for i in range(len(ast))]
    if ast.op == 'RotateRight':
        if ast.args[1].symbolic:
            print('what?')
        else:
            return [children[0][(i + ast.args[1].args[0]) % len(ast.args[0])] for i in range(len(ast))]
    if ast.op == 'RotateLeft':
        if ast.args[1].symbolic:
            print('what?')
        else:
            return [children[0][(i - ast.args[1].args[0]) % len(ast.args[0])] for i in range(len(ast))]

    if ast.op == 'Concat':
        out = []
        for child in reversed(children):
            out.extend(child)
        return out

    if ast.op not in complained:
        complained.add(ast.op)
        print("unimplemented", ast.op)

    # final case: mix all bits
    all_bits = mix_all_bits(children)
    if type(ast) is claripy.ast.bv.BV:
        return [all_bits] * len(ast)
    elif type(ast) is claripy.ast.bool.Bool:
        return [all_bits]
    else:
        import ipdb; ipdb.set_trace()
        return None

complained = set()

# mix masks are internal nodes to the AST where we have identified that there are more bits of input than output
mix_masks = {} # map from mixmask to (original translation, root bits, height) where height is the max height of all its masked bits + 1

def classify_bits(bits):
    masks = set()
    free_bits = set()

    if type(bits) is bool:
        return masks, free_bits

    for bit in bits:
        mask = bit[0]
        if mask in mix_masks:
            masks.add(mask)
        else:
            free_bits.add(bit)

    return masks, free_bits

def unmask(bits, mask):
    if type(bits) is bool:
        return bits

    result = set()
    for bit in bits:
        mmask, i = bit
        if mmask == mask:
            result.update(mix_masks[mask][0][i])
        else:
            result.add(bit)
    return result

def manage_mix_masks(ast, val):
    # check: do we have interdependent mix masks?
    masks, free_bits = classify_bits(mix_bits(val))

    counts = Counter()
    for mask in masks:
        counts.update(mix_masks[mask][1])

    # remove masks (in order of recency) until there are no more conflicts
    while True:
        conflicts = [bit for bit, count in counts.items() if count > 1 or (count > 0 and bit in free_bits)]
        if not conflicts:
            break

        try:
            removeme = max((m for m in masks if mix_masks[m][1].intersection(conflicts)), key=lambda m: mix_masks[m][2])
        except ValueError:
            import ipdb; ipdb.set_trace()

        if removeme.ast.op == 'SMod':
            import ipdb; ipdb.set_trace()
        masks.remove(removeme)
        counts.subtract(mix_masks[removeme][1])
        val = [unmask(bits, removeme) for bits in val]
        new_masks, new_free_bits = classify_bits(mix_bits(mix_masks[removeme][0]))
        for mask in new_masks:
            if mask not in masks:
                masks.add(mask)
                counts.update(mix_masks[mask][1])
        free_bits.update(new_free_bits)

    # check: should we mix mask now?
    max_bits = len([x for x in val if type(x) is not bool])
    mixed = mix_bits(val)

    if len(mixed) > max_bits:
        # it's time for a MIX MASK
        oldval = val
        if type(ast) is claripy.ast.Bool:
            newval = [{(ast.cache_key, 0)}]
        else:
            newval = [{(ast.cache_key, i)} if type(oldval[i]) is not bool else oldval[i] for i in range(len(ast))]
        val = newval

        max_height = 0
        roots = set()
        for bit in mixed:
            mask = bit[0]
            if mask in mix_masks:
                info = mix_masks[mask]
                max_height = max(info[2], max_height)
                roots.update(info[1])
            else:
                roots.add(bit)

        mix_masks[ast.cache_key] = (oldval, roots, max_height + 1)

    return val


def mix_bits(bits):
    """
    Given a list of bit result values (valid entries-in-lists as output by interpret()), mix all
    the taints together.
    """
    return set().union(*(bit for bit in bits if type(bit) is not bool))

def mix_all_bits(children):
    """
    Given a list of results from interpret(), mix all the taints together.
    """
    return set().union(*(mix_bits(child) for child in children))

def bake_masks(in_bits):
    replacements = {mask: claripy.BVS('mask', len(mask.ast)) for mask in mix_masks if type(mask.ast) is not claripy.ast.Bool}
    return [bit if type(bit) is bool else {(mask.ast.replace_dict(replacements).cache_key, i) for mask, i in bit} for bit in in_bits]

def convert_to_bools(in_bits):
    return [bit if type(bit) is bool else {mask if type(mask.ast) is claripy.ast.Bool else (mask.ast[i] == 1).cache_key for mask, i in bit} for bit in in_bits]

def simplify(in_bits):
    """
    Given a bit result value (valid entries-in-lists as output by interpret()), simplify the
    result to remove bits which are implied by other bits.
    """
    partitions = defaultdict(list)
    completeness_tracker = defaultdict(set)
    complete_vars = set()

    out = set()

    # if all bits of a variable are individually constrained, we can safely discard all other
    # assertions about it
    for bitkey in in_bits:
        if type(bitkey) is bool:
            continue
        bit = bitkey.ast

        # (also: weed out bits which have effectively 0 entropy)
        if bit.op in ('__eq__', '__ne__') and len(bit.args[0]) > 20 and not (bit.args[0].symbolic and bit.args[1].symbolic):
            continue

        partitions[bit.variables].append(bitkey)

        if len(bit.variables) == 1:
            var = next(iter(bit.variables))
            varsize = int(var.split('_')[-1]) # :(

            if bit.op in ('__eq__', '__ne__') and bit.args[1].op == 'BVV' and bit.args[0].op == 'Extract' and bit.args[0].args[0] == bit.args[0].args[1] and bit.args[0].args[2].op == 'BVS':
                completeness_tracker[var].add(bitkey)
            if bit.op == 'BoolS':
                completeness_tracker[var].add(bitkey)

            if len(completeness_tracker[var]) >= varsize:
                complete_vars.add(var)
                out.update(completeness_tracker[var])

    for varset in partitions:
        if all(var in complete_vars for var in varset):
            continue

        bits = list(partitions[varset])
        #i = 0
        #while i < len(bits):
        #    bitA = bits[i]
        #    j = 0
        #    while j < len(bits):
        #        if j == i:
        #            continue
        #        bitB = bits[j]

        #        # check: does B add any information over A?
        #        # IDK HOW TO DO THIS
        #        # ... two days of information theory deep dive later: FUCK THAT'S A LOT

        #        j += 1
        #    i += 1

        out.update(bits)

    return out

def compute_final(in_bits):
    entropy = information.H1([x.ast for x in in_bits if type(x) is not bool])

    all_vars = set()
    for bitkey in in_bits:
        if type(bitkey) is bool:
            continue
        all_vars.update(bitkey.ast.variables)
    sanity_check = 0
    for var in all_vars:
        sanity_check += int(var.split('_')[-1])

    if sanity_check < entropy:
        l.warning('Sanity check failure: computed %s but only %s available. falling back', entropy, sanity_check)
        return sanity_check

    return entropy
