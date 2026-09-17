"""Counting and exhaustive checks for two-region ternary encodings (see research/region-memory.md)."""

import itertools
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


states = (-1, 0, 1)
LOG2_3 = math.log2(3)
checks = []
tables = {}
rng = random.Random(20260912)


def verify(name, condition, detail=None):
    if not condition:
        raise AssertionError(f'{name}: {detail}' if detail is not None else name)
    checks.append(name)


def verify_all(name, cases):
    """cases yields (inputs, left, right); the first mismatch is reported with its inputs and both sides."""
    for inputs, left, right in cases:
        if left != right:
            raise AssertionError(f'{name}: inputs={inputs!r} left={left!r} right={right!r}')
    checks.append(name)


def verify_equal(name, left, right):
    if left != right:
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            index = next(i for i, (a, b) in enumerate(zip(left, right)) if a != b)
            raise AssertionError(f'{name}: first mismatch at index {index}: left={left[index]!r} right={right[index]!r}')
        raise AssertionError(f'{name}: left={left!r} right={right!r}')
    checks.append(name)


def binary_entropy(probability):
    return sum(-q * math.log2(q) for q in (probability, 1 - probability) if q > 0)


def ternary_entropy(density, negative_share=0.5):
    """Bits per weight for independent trits: P(nonzero)=density, P(-1 | nonzero)=negative_share."""
    return binary_entropy(density) + density * binary_entropy(negative_share)


def random_trits(count, density, source=None):
    source = source or rng
    return [source.choice((-1, 1)) if source.random() < density else 0 for _ in range(count)]


def dot(weights, inputs):
    return sum(w * x for w, x in zip(weights, inputs))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


# 1. The information budget and the fixed-layout encodings.
densities = (1.0, 2 / 3, 0.5, 0.3, 0.1)
block_bits, block_count_bits = 256, 16                # 16-bit count per block, relative to its superblock
superblock_bits, superblock_count_bits = 65536, 32    # 32-bit absolute count per superblock
rank_directory = block_count_bits / block_bits + superblock_count_bits / superblock_bits
tables['bits_per_weight'] = [
    {
        'density': round(d, 4),
        'entropy_bound': round(ternary_entropy(d), 4),
        'fixed_2bit': 2.0,
        'five_trits_per_byte': 1.6,
        'pos_neg_planes': 2.0,
        'mask_plus_signs': round(1 + d, 4),
        'mask_plus_signs_with_rank_directory': round(1 + d + rank_directory, 4),
        'unsigned_digits_plus_zero_point': 2.0,
    }
    for d in densities
]
tables['rank_directory_bits_per_weight'] = round(rank_directory, 4)
verify('uniform ternary entropy is log2(3)', abs(ternary_entropy(2 / 3) - LOG2_3) < 1e-12, f'{ternary_entropy(2 / 3)} vs {LOG2_3}')
verify('five trits fit one byte', 3 ** 5 == 243 <= 256 and 8 / 5 == 1.6)
verify_all('mask+sign planes cost one plus density and never beat the entropy bound', ((d, 1 + d >= ternary_entropy(d) - 1e-12, True) for d in densities))
verify_all('mask+sign planes meet the entropy bound only at density one half', ((d, abs(1 + d - ternary_entropy(d)) < 1e-12, d == 0.5) for d in densities))
verify_all('mask plane overhead equals one minus the binary entropy of the density', ((d, abs((1 + d - ternary_entropy(d)) - (1 - binary_entropy(d))) < 1e-12, True) for d in densities))

# 2. Which of the six state permutations can be undone on the input side of a linear layer?
permutations = tuple(itertools.permutations(states))
# For an affine map a*x+b through the three states, a = t(1) - t(0) lies in -2..2 and b = t(0) in -1..1, so -4..4 is exhaustive.
affine = sorted(outputs for outputs in permutations if any(all(a * x + b == dict(zip(states, outputs))[x] for x in states) for a in range(-4, 5) for b in range(-4, 5)))
verify_equal('exactly identity and sign flip are affine permutations of the three states', affine, sorted([(-1, 0, 1), (1, 0, -1)]))
small_inputs = range(-3, 4)
compensable = sorted(outputs for outputs in permutations if any(all(dict(zip(states, outputs))[w] * (c * x) == w * x for w in states for x in small_inputs) for c in (-1, 1)))
verify_equal('exactly identity and sign flip are compensable by an input sign', compensable, affine)
verify_all('every non-affine permutation moves zero', ((outputs, dict(zip(states, outputs))[0] != 0, True) for outputs in permutations if outputs not in affine))

# 3. Dot-product identities that let a kernel act on regions without materializing weights.
trials = []
for _ in range(300):
    length = rng.randint(1, 40)
    trials.append((random_trits(length, rng.random()), [rng.randint(-50, 50) for _ in range(length)]))
verify_all('unsigned digits plus one zero-point correction reproduce the dot product', (((w, x), dot([a + 1 for a in w], x) - sum(x), dot(w, x)) for w, x in trials))
verify_all('unsigned digits lie in 0..2', ((a, a + 1 in (0, 1, 2), True) for w, _ in trials for a in w))
verify_all('positive and negative planes reproduce the dot product', (((w, x), sum(b for a, b in zip(w, x) if a == 1) - sum(b for a, b in zip(w, x) if a == -1), dot(w, x)) for w, x in trials))
verify_all('mask and sign planes reproduce the dot product', (((w, x), sum(b for a, b in zip(w, x) if a != 0) - 2 * sum(b for a, b in zip(w, x) if a == -1), dot(w, x)) for w, x in trials))


# 4. A mask region with a two-level rank directory gives random access into a compacted sign region.
def build_rank_directory(bits):
    """Absolute 32-bit count per superblock; 16-bit count per block relative to its superblock."""
    superblocks, blocks, running = [], [], 0
    for start in range(0, len(bits), block_bits):
        if start % superblock_bits == 0:
            superblocks.append(running)
        blocks.append(running - superblocks[-1])
        running += sum(bits[start:start + block_bits])
    return superblocks, blocks


def rank(bits, directory, position):
    """Set bits strictly before position: superblock count + block count + in-block popcount."""
    superblocks, blocks = directory
    return superblocks[position // superblock_bits] + blocks[position // block_bits] + sum(bits[position - position % block_bits:position])


def directory_bits(directory):
    superblocks, blocks = directory
    return len(superblocks) * superblock_count_bits + len(blocks) * block_count_bits


count = 4096
weights = random_trits(count, 0.4)
mask = [1 if w != 0 else 0 for w in weights]
signs = [w for w in weights if w != 0]
directory = build_rank_directory(mask)
verify_all('rank-indexed sign plane reproduces every weight', ((j, signs[rank(mask, directory, j)] if mask[j] else 0, weights[j]) for j in range(count)))
rank_bits = {'mask_bits': count, 'sign_bits': len(signs), 'directory_bits': directory_bits(directory)}
tables['rank_example'] = {'weights': count, 'nonzeros': len(signs), **rank_bits, 'bits_per_weight': round(sum(rank_bits.values()) / count, 4), 'entropy_bound': round(ternary_entropy(len(signs) / count), 4)}
boundary_source = random.Random(1)
long_mask = [1 if boundary_source.random() < 0.9 else 0 for _ in range(superblock_bits + 3000)]
long_directory = build_rank_directory(long_mask)
probe_positions = sorted({0, block_bits - 1, superblock_bits - 1, superblock_bits, superblock_bits + 1, len(long_mask) - 1, *(boundary_source.randrange(len(long_mask)) for _ in range(60))})
verify_all('two-level directory ranks correctly across a superblock boundary with block counts under sixteen bits', (((j, max(long_directory[1])), rank(long_mask, long_directory, j) if max(long_directory[1]) < 2 ** block_count_bits else None, sum(long_mask[:j])) for j in probe_positions))

# 5. An escape code that points into an exception region: spare fourth code versus the numerical zero.
outlier_share, zero_share = 0.01, 0.30
count = 10000
original = []
for _ in range(count):
    draw = rng.random()
    if draw < outlier_share:
        original.append(rng.choice([v for v in range(-127, 128) if abs(v) > 1]))
    elif draw < outlier_share + zero_share:
        original.append(0)
    else:
        original.append(rng.choice((-1, 1)))
ESCAPE = 3
codes = [ESCAPE if abs(v) > 1 else v for v in original]
exceptions_a = [v for v in original if abs(v) > 1]
escape_mask = [1 if c == ESCAPE else 0 for c in codes]
directory_a = build_rank_directory(escape_mask)
decoded_a = [exceptions_a[rank(escape_mask, directory_a, j)] if codes[j] == ESCAPE else codes[j] for j in range(count)]
verify_equal('spare fourth code as an escape pointer round-trips', decoded_a, original)
main_b = [v if v in (-1, 1) else 0 for v in original]
exceptions_b = [v for v in original if v not in (-1, 1)]
zero_mask = [1 if t == 0 else 0 for t in main_b]
directory_b = build_rank_directory(zero_mask)
decoded_b = [exceptions_b[rank(zero_mask, directory_b, j)] if main_b[j] == 0 else main_b[j] for j in range(count)]
verify_equal('trit zero as an escape pointer round-trips', decoded_b, original)
cost_a = 2 * count + 8 * len(exceptions_a) + directory_bits(directory_a)
cost_b = LOG2_3 * count + 8 * len(exceptions_b) + directory_bits(directory_b)
tables['escape_example'] = {'weights': count, 'outliers': len(exceptions_a), 'true_zeros': len(exceptions_b) - len(exceptions_a), 'spare_code_bits_per_weight': round(cost_a / count, 4), 'zero_as_pointer_bits_per_weight': round(cost_b / count, 4), 'fourth_code_penalty_over_packed_ternary': round(2 - LOG2_3, 4)}
verify('using the numerical zero as a pointer costs more than the spare code when true zeros are common', cost_b > cost_a, f'{cost_b} vs {cost_a}')

# 6. An activation-side lookup table turns a packed weight group into a table index.
group = 3
weights = random_trits(12, 2 / 3)
inputs = [rng.randint(-9, 9) for _ in range(12)]
looked_up = 0
for start in range(0, 12, group):
    block_inputs = inputs[start:start + group]
    table = {pattern: dot(pattern, block_inputs) for pattern in itertools.product(states, repeat=group)}
    looked_up += table[tuple(weights[start:start + group])]
verify_equal('activation-side lookup table replaces multiply-accumulate', looked_up, dot(weights, inputs))
verify('two-trit and three-trit lookup indices fit four and five bits', 3 ** 2 == 9 <= 16 and 3 ** 3 == 27 <= 32)

# 7. Template dictionaries on random blocks, priced with a flat index and log2(3) per dictionary trit (pessimistic for the dictionary).
dictionary_rows = []
for density in (2 / 3, 0.2):
    for block in (2, 4, 8, 16):
        total = 100000 - 100000 % block
        blocks = [tuple(w) for w in zip(*[iter(random_trits(total, density))] * block)]
        distinct = len(set(blocks))
        index_bits = math.ceil(math.log2(distinct)) / block if distinct > 1 else 0.0
        dictionary_bits_per_weight = distinct * block * LOG2_3 / total
        dictionary_rows.append({'density': round(density, 4), 'block': block, 'weights': total, 'blocks': total // block, 'distinct': distinct, 'possible': 3 ** block, 'index_bits_per_weight': round(index_bits, 4), 'dictionary_bits_per_weight': round(dictionary_bits_per_weight, 4), 'total_bits_per_weight': round(index_bits + dictionary_bits_per_weight, 4), 'entropy_bound': round(ternary_entropy(density), 4)})
tables['dictionary_on_random_blocks'] = dictionary_rows
verify_all('flat-index dictionary totals never fall below log2(3) on uniform random blocks in this sample', (((r['density'], r['block']), r['total_bits_per_weight'] >= LOG2_3 - 1e-9, True) for r in dictionary_rows if r['density'] == round(2 / 3, 4)))
verify_all('flat-index dictionary totals never fall below the entropy bound on sparse random blocks in this sample', (((r['density'], r['block']), r['total_bits_per_weight'] >= r['entropy_bound'] - 1e-9, True) for r in dictionary_rows if r['density'] == 0.2))


# 8. Under row-and-column compensation, an undecoded per-block sign flip must factor as row signs times column signs.
def factors_into_row_and_column_signs(flip):
    rows, columns = len(flip), len(flip[0])
    return any(all(flip[i][j] == r[i] * c[j] for i in range(rows) for j in range(columns)) for r in itertools.product((-1, 1), repeat=rows) for c in itertools.product((-1, 1), repeat=columns))


verify('a rank-one sign pattern factors into row and column signs', factors_into_row_and_column_signs([[1, -1, 1], [-1, 1, -1]]))
verify('a non-rank-one block sign pattern has no row/column compensation', not factors_into_row_and_column_signs([[1, 1], [1, -1]]))
matrix, vector, row_signs, column_signs = [[-1, 0, 1], [1, -1, 1]], [2, -3, 4], [1, -1], [-1, 1, 1]
stored = [[matrix[i][j] * row_signs[i] * column_signs[j] for j in range(3)] for i in range(2)]
recovered = [row_signs[i] * sum(stored[i][j] * (column_signs[j] * vector[j]) for j in range(3)) for i in range(2)]
verify_equal('row and column sign compensation reproduces the matrix-vector product', recovered, matvec(matrix, vector))
tables['selector_bits'] = {'full_permutation_selector_ideal_bits_per_block': round(math.log2(6), 4), 'full_permutation_selector_flat_bits_per_block': 3, 'sign_flip_selector_bits_per_block': 1, 'per_weight_at_block_16': {'ideal_full': round(math.log2(6) / 16, 4), 'flat_full': round(3 / 16, 4), 'flip': round(1 / 16, 4)}}

result = {
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'passed',
    'checks': checks,
    'tables': tables,
    'random_seed': 20260912,
    'limitations': [
        'Entropy bounds assume independent identically distributed trits with balanced signs; trained weights need measured statistics.',
        'Dictionary rows are one seeded sample on random data priced with a flat index and log2(3) per dictionary trit; they are not a proof and say nothing about trained weights.',
        'Bit counts exclude scales, padding, alignment and any accumulator; no kernel was timed.',
    ],
}
result_path = Path('outputs/region-encoding.json')
result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'check_count': len(checks)}))
