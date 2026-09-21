"""Exhaustive checks on the symmetry group of the balanced-ternary carrier S = {-1, 0, +1}.

Companion to research/carrier-symmetry.md. Standard library only. Run from the project root:

    python experiments/verify_carrier_symmetry.py

Writes outputs/carrier-symmetry.json. A failing check names the assertion, the first failing
input and both sides of the equality (the diagnostic contract asked for in
research/independent-review.md). Domain: the three states only; the three operator families
of research/operator-reference.md (ordered logic min/max, signed product, addition modulo 3)
are kept distinct throughout. No claim here concerns model quality, memory or speed.
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path


S = (-1, 0, 1)
pairs = tuple(itertools.product(S, repeat=2))
checks = []


def bal(n):
    """The member of S congruent to n modulo 3 (research/operator-reference.md section 3)."""
    return (n + 1) % 3 - 1


def verify(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def verify_all(name, inputs, left, right):
    """left(i) == right(i) for every input; on failure report the first input and both sides."""
    for i in inputs:
        l, r = left(i), right(i)
        if l != r:
            raise AssertionError(f'{name}: at input {i!r} left={l!r} right={r!r}')
    checks.append(name)


# --- the six reversible unary maps, named as in research/operator-reference.md section 3 ---

def N(c):
    return lambda x: bal(c - x)


perms = {
    'identity': lambda x: x,
    'N_0': N(0),          # sign reversal: swap -1 and +1, fix 0
    'N_+1': N(1),         # swap 0 and +1, fix -1
    'N_-1': N(-1),        # swap -1 and 0, fix +1
    'cycle_plus': lambda x: bal(x + 1),
    'cycle_minus': lambda x: bal(x - 1),
}
table = {name: tuple(f(x) for x in S) for name, f in perms.items()}


def compose(f, g):
    """f after g."""
    return lambda x: f(g(x))


def name_of(f):
    out = tuple(f(x) for x in S)
    for name, t in table.items():
        if t == out:
            return name
    raise KeyError(out)


# 1. the named maps are exactly the six permutations of S
verify('the six named maps are bijections of S', all(sorted(t) == list(S) for t in table.values()))
verify('the six named maps are distinct', len(set(table.values())) == 6)
verify('they exhaust the 3! permutations', set(table.values()) == set(itertools.permutations(S)))

# 2. orders: identity and the three N_c are involutions; the two cycles have order 3
verify('N_0, N_+1, N_-1 are involutions', all(all(perms[n](perms[n](x)) == x for x in S) for n in ('N_0', 'N_+1', 'N_-1')))
verify('cycle_plus and cycle_minus have order 3', all(name_of(compose(perms[n], compose(perms[n], perms[n]))) == 'identity' and name_of(compose(perms[n], perms[n])) != 'identity' for n in ('cycle_plus', 'cycle_minus')))
verify('exactly four involutions including identity', sum(1 for f in perms.values() if all(f(f(x)) == x for x in S)) == 4)

# 3. order: only N_0 reverses the order -1 < 0 < +1; only identity preserves it
order_reversing = [n for n, f in perms.items() if all((f(x) > f(y)) == (x < y) for x, y in pairs if x != y)]
order_preserving = [n for n, f in perms.items() if all((f(x) < f(y)) == (x < y) for x, y in pairs if x != y)]
verify('N_0 is the only order-reversing bijection', order_reversing == ['N_0'])
verify('identity is the only order-preserving bijection', order_preserving == ['identity'])

# 4. closure under composition: a group of order 6
composition = {a: {b: name_of(compose(perms[a], perms[b])) for b in perms} for a in perms}
verify('composition is closed (a group)', all(v in perms for row in composition.values() for v in row.values()))
verify('identity is the neutral element', all(composition['identity'][b] == b and composition[b]['identity'] == b for b in perms))
verify('every element has an inverse', all(any(composition[a][b] == 'identity' for b in perms) for a in perms))
verify('the group is non-abelian', composition['N_0']['N_-1'] != composition['N_-1']['N_0'])

# 5. the composition law of the three swaps: N_c after N_d is the shift by c - d
verify_all('N_c after N_d equals x -> bal(c - d + x)', tuple(itertools.product((0, 1, -1), (0, 1, -1), S)),
           lambda t: bal(t[0] - bal(t[1] - t[2])), lambda t: bal(t[0] - t[1] + t[2]))

# 6. the two identities of the overlap note, in this notation
verify_all('N_0 after N_-1 equals cycle_plus', S, compose(perms['N_0'], perms['N_-1']), perms['cycle_plus'])
verify_all('N_+1 after N_0 equals cycle_plus', S, compose(perms['N_+1'], perms['N_0']), perms['cycle_plus'])
verify('every reflection pair composes to a rotation', all(composition[a][b] in ('identity', 'cycle_plus', 'cycle_minus') for a in ('N_0', 'N_+1', 'N_-1') for b in ('N_0', 'N_+1', 'N_-1')))

# 7. dihedral presentation: r^2 = 1, s^3 = 1, r s r = s^-1, and <r, s> is the whole group
verify('N_0 conjugates cycle_plus to cycle_minus', name_of(compose(perms['N_0'], compose(perms['cycle_plus'], perms['N_0']))) == 'cycle_minus')
generated = {'identity'}
frontier = {'identity'}
while frontier:
    new = {composition[g][h] for g in frontier for h in ('N_0', 'cycle_plus')} - generated
    generated |= new
    frontier = new
verify('N_0 and cycle_plus generate all six (the dihedral group of the triangle, order 6)', generated == set(perms))

# 8. which maps fix zero
moves_zero = {n: perms[n](0) != 0 for n in perms}
verify('exactly identity and N_0 fix zero', {n for n, m in moves_zero.items() if not m} == {'identity', 'N_0'})

# 9. the commutation table: each map against each operator family, families kept distinct
ops = {
    'min (ordered AND)': min,
    'max (ordered OR)': max,
    'signed product (the XOR-labelled table)': lambda x, y: x * y,
    'addition modulo 3 (balanced)': lambda x, y: bal(x + y),
}
dual = {'min (ordered AND)': max, 'max (ordered OR)': min}


def classify(f, op, name):
    auto = all(f(op(x, y)) == op(f(x), f(y)) for x, y in pairs)
    anti = name in dual and all(f(op(x, y)) == dual[name](f(x), f(y)) for x, y in pairs)
    push = all(f(op(x, y)) == op(f(x), y) for x, y in pairs) and all(f(op(x, y)) == op(x, f(y)) for x, y in pairs)
    return {'automorphism': auto, 'anti_automorphism': anti, 'pushable_onto_one_input': push}


commutation = {opname: {pname: classify(f, op, opname) for pname, f in perms.items()} for opname, op in ops.items()}


def those(opname, key):
    return {p for p, c in commutation[opname].items() if c[key]}


verify('min and max: identity is the only automorphism', those('min (ordered AND)', 'automorphism') == {'identity'} and those('max (ordered OR)', 'automorphism') == {'identity'})
verify('min and max: N_0 is the only anti-automorphism (De Morgan)', those('min (ordered AND)', 'anti_automorphism') == {'N_0'} and those('max (ordered OR)', 'anti_automorphism') == {'N_0'})
verify('signed product: identity is the only automorphism', those('signed product (the XOR-labelled table)', 'automorphism') == {'identity'})
verify('signed product: exactly identity and N_0 can be pushed onto one input', those('signed product (the XOR-labelled table)', 'pushable_onto_one_input') == {'identity', 'N_0'})
verify('addition modulo 3: exactly identity and N_0 are automorphisms', those('addition modulo 3 (balanced)', 'automorphism') == {'identity', 'N_0'})
verify('addition modulo 3: exactly identity and the two cycles can be pushed onto one input', those('addition modulo 3 (balanced)', 'pushable_onto_one_input') == {'identity', 'cycle_plus', 'cycle_minus'})
verify('the zero-moving maps have no property under min, max or the signed product', all(not any(commutation[o][p].values()) for o in list(ops)[:3] for p in ('N_+1', 'N_-1', 'cycle_plus', 'cycle_minus')))

# 10. agreement with research/region-memory.md claim 8 (identity and sign flip are the maps an input sign undoes)
verify('agrees with region-memory claim 8: the product-pushable set is {identity, N_0}', those('signed product (the XOR-labelled table)', 'pushable_onto_one_input') == {n for n, m in moves_zero.items() if not m})

# 11. hardware reading: the differential conductance pair (research/carrier-symmetry.md section 3b)
# A weight w in S is stored as two binary cells (g_plus, g_minus) with w = g_plus - g_minus. This is the
# finite code only: a "write" is a change of one cell's state; endurance, drift, energy and analog
# non-idealities are not modelled and no claim about them is made.
pair = {-1: (0, 1), 0: (0, 0), 1: (1, 0)}
decode = {code: w for w, code in pair.items()}
verify('differential pair decodes as g_plus minus g_minus', all(pair[w][0] - pair[w][1] == w for w in S))
verify('zero is the only code with no conducting cell', [w for w in S if pair[w] == (0, 0)] == [0])

write_free_pair_ops = {'identity': lambda g: g, 'swap_lines': lambda g: (g[1], g[0])}
induced = {n: name_of(lambda x, f=f: decode[f(pair[x])]) for n, f in write_free_pair_ops.items()}
verify('swapping the two lines induces exactly N_0; no other map is write-free', induced == {'identity': 'identity', 'swap_lines': 'N_0'})


def reprogram_cost(f):
    """Cell-state changes needed to turn every stored code pair[x] into pair[f(x)], summed over S."""
    return sum(int(pair[f(x)][i] != pair[x][i]) for x in S for i in (0, 1))


reprogram = {n: reprogram_cost(f) for n, f in perms.items()}
verify('reprogram costs: identity 0, N_0 4, the two zero-moving swaps 2 each, the two cycles 4 each',
       reprogram == {'identity': 0, 'N_0': 4, 'N_+1': 2, 'N_-1': 2, 'cycle_plus': 4, 'cycle_minus': 4})
zero_current_preserved = {n: perms[n](0) == 0 for n in perms}
verify('the non-conducting code stays on the zero weight exactly under the zero-fixing maps',
       {n for n, ok in zero_current_preserved.items() if ok} == {'identity', 'N_0'})
verify('the write-free maps are exactly the product-pushable maps',
       set(induced.values()) == those('signed product (the XOR-labelled table)', 'pushable_onto_one_input'))

result = {
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'passed',
    'domain': 'S = {-1, 0, +1}; every check is exhaustive over S, S x S or the stated index set',
    'checks': checks,
    'permutations': {n: list(t) for n, t in table.items()},
    'composition_table_row_after_column': composition,
    'moves_zero': moves_zero,
    'commutation_table': commutation,
    'differential_pair': {
        'code': {str(w): list(c) for w, c in pair.items()},
        'write_free_pair_ops_induce': induced,
        'reprogram_cost_cell_writes_over_S': reprogram,
        'zero_current_code_preserved': zero_current_preserved,
    },
}
result_path = Path('outputs/carrier-symmetry.json')
result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'check_count': len(checks)}))
