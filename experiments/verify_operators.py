"""Exhaustive checks for the kickoff's finite three-state operator claims."""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path


states = (-1, 0, 1)
pairs = tuple(itertools.product(states, repeat=2))
triples = tuple(itertools.product(states, repeat=3))
checks = []


def verify(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def balanced(n):
    return (n + 1) % 3 - 1


def matvec(matrix, vector):
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


verify('supplied NOT equals sign inversion', [-x for x in states] == [1, 0, -1])
verify('supplied AND equals min', [[min(x, y) for y in states] for x in states] == [[-1, -1, -1], [-1, 0, 0], [-1, 0, 1]])
verify('printed OR table equals max', [[max(x, y) for y in states] for x in states] == [[-1, 0, 1], [0, 0, 1], [1, 1, 1]])
verify('supplied XOR-labelled table equals signed multiplication', [[x*y for y in states] for x in states] == [[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
verify('printed XOR table equals negative signed product', [[-x*y for y in states] for x in states] == [[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
verify('strong-Kleene XOR construction equals negative product', all(min(max(x, y), -min(x, y)) == -x*y for x, y in pairs))
verify('reference XOR expression equals negative product', all(max(min(x, -y), min(-x, y)) == -x*y for x, y in pairs))
verify('De Morgan min-to-max', all(-min(x, y) == max(-x, -y) for x, y in pairs))
verify('De Morgan max-to-min', all(-max(x, y) == min(-x, -y) for x, y in pairs))
verify('min and max associative', all(min(min(x, y), z) == min(x, min(y, z)) and max(max(x, y), z) == max(x, max(y, z)) for x, y, z in triples))
verify('min and max mutually distributive', all(min(x, max(y, z)) == max(min(x, y), min(x, z)) and max(x, min(y, z)) == min(max(x, y), max(x, z)) for x, y, z in triples))
verify('min and max commutative, identities and absorption', all(min(x, y) == min(y, x) and max(x, y) == max(y, x) and min(x, max(x, y)) == x and max(x, min(x, y)) == x for x, y in pairs) and all(min(x, 1) == x and max(x, -1) == x for x in states))
verify('logical endpoints match Boolean truth tables', all((min(x, y) == 1) == ((x == 1) and (y == 1)) and (max(x, y) == 1) == ((x == 1) or (y == 1)) and (-x*y == 1) == ((x == 1) != (y == 1)) and (x*y == 1) == ((x == 1) == (y == 1)) for x, y in itertools.product((-1, 1), repeat=2)))
verify('XOR is commutative and associative', all(-x*y == -y*x for x, y in pairs) and all(-(-x*y)*z == -x*(-y*z) for x, y, z in triples))
verify('unknown excluded middle remains unknown', max(0, -0) == 0)
verify('unknown contradiction expression remains unknown', min(0, -0) == 0)
verify('ordinary addition is not closed', 1 + 1 not in states)
verify('printed mod3 addition table', [[balanced(x+y) for y in states] for x in states] == [[1, -1, 0], [-1, 0, 1], [0, 1, -1]])
verify('printed mod3 subtraction table', [[balanced(x-y) for y in states] for x in states] == [[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
verify('mod3 addition is closed and associative', all(balanced(x+y) in states for x, y in pairs) and all(balanced(balanced(x+y)+z) == balanced(x+balanced(y+z)) for x, y, z in triples))
verify('mod3 distributivity', all(balanced(x*balanced(y+z)) == balanced(x*y+x*z) for x, y, z in triples))
verify('mod3 multiplicative associativity, identities and inverses', all((x*y)*z == x*(y*z) for x, y, z in triples) and all(balanced(x+0) == x and balanced(x-x) == 0 and x*1 == x for x in states) and all(any(x*y == 1 for y in states) for x in (-1, 1)))
verify('balanced addition preserves carry', all((x+y-balanced(x+y)) % 3 == 0 and (x+y-balanced(x+y))//3 in states and x+y == balanced(x+y)+3*((x+y-balanced(x+y))//3) for x, y in pairs))
verify('mod3 differs from ordinary dot product', balanced(1*1+1*1) == -1 and 1*1+1*1 == 2)

permutations = tuple(itertools.permutations(states))
involutions = []
for outputs in permutations:
    function = dict(zip(states, outputs))
    if all(function[function[x]] == x for x in states):
        involutions.append(outputs)
nontrivial = set(involutions) - {states}
swaps = {tuple(balanced(c-x) for x in states) for c in states}
verify('three nontrivial involutive permutations', len(nontrivial) == 3 and nontrivial == swaps)
verify('only sign inversion reverses the order bijectively', [p for p in permutations if all(p[i] > p[i+1] for i in range(2))] == [(1, 0, -1)])
verify('27 unary functions and six permutations', len(tuple(itertools.product(states, repeat=3))) == 27 and len(permutations) == 6)
verify('two cyclic shifts have order three', all(all(balanced(balanced(balanced(x+c)+c)+c) == x for x in states) and any(balanced(balanced(x+c)+c) != x for x in states) for c in (-1, 1)))
verify('cyclic shifts undo each other', all(balanced(balanced(x+1)-1) == x and balanced(balanced(x-1)+1) == x for x in states))

matrix = [[-1, 0, 1], [1, -1, 1]]
vector = [2, -3, 4]
signs = [-1, 1, -1]
changed_matrix = [[w*s for w, s in zip(row, signs)] for row in matrix]
changed_vector = [x*s for x, s in zip(vector, signs)]
verify('paired column and input sign changes preserve a matrix-vector product', matvec(matrix, vector) == matvec(changed_matrix, changed_vector))
permutation = [2, 0, 1]
verify('paired column and input permutations preserve a matrix-vector product', matvec(matrix, vector) == matvec([[row[j] for j in permutation] for row in matrix], [vector[j] for j in permutation]))
verify('unpaired representation change can change a result', matvec(matrix, vector) != matvec(changed_matrix, vector))
verify('a zero-moving swap can change ordinary dot products', 0*1 != balanced(1-0)*1)
verify('sign reversal does not commute with ReLU', max(0, -1) != -max(0, 1))
verify('signed dot product is positive sum minus negative sum', all(sum(w*x for w, x in zip(row, vector)) == sum(x for w, x in zip(row, vector) if w == 1) - sum(x for w, x in zip(row, vector) if w == -1) for row in matrix))

result = {
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'passed',
    'checks': checks,
    'finite_scope': {'states': 3, 'pairs_per_binary_law': 9, 'triples_per_ternary_law': 27, 'permutations': 6},
    'limitations': ['Matrix identity checked on one example; the general proof is algebraic.', 'No neural model accuracy, timing, memory benchmark, hardware runtime, or ecosystem service test was performed.'],
}
result_path = Path('outputs/math-verification.json')
result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'check_count': len(checks)}))
