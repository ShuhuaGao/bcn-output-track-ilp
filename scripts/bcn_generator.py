import random
import json
import math

class BoolInt(int):
    def __new__(cls, value):
        return super().__new__(cls, 1 if value else 0)

    def __invert__(self):
        return BoolInt(1 - self)

    def __and__(self, other): return BoolInt(super().__and__(other))
    def __or__(self, other): return BoolInt(super().__or__(other))
    def __xor__(self, other): return BoolInt(super().__xor__(other))
    
    def __rand__(self, other): return BoolInt(other & self)
    def __ror__(self, other): return BoolInt(other | self)
    def __rxor__(self, other): return BoolInt(other ^ self)

DEGREE_CONFIG = {
    2: 40,  
    3: 40,  
    4: 15,  
    6: 5    
}

def evaluate_logic(rule_str, state_dict):
    """Implement the evaluate logic operation.
    
    Args:
    rule_str: Input used by this operation.
    state_dict: Input used by this operation.
    """
    try:
        safe_state = {k: BoolInt(v) for k, v in state_dict.items()}
        val = eval(rule_str, {}, safe_state)
        return int(val)
    except Exception as e:
        # print(f"Logic Eval Error: {e} in {rule_str}")
        return 0

def generate_logic_rule(allowed_vars):
    """Implement the generate logic rule operation.
    
    Args:
    allowed_vars: Input used by this operation.
    """
    options = list(DEGREE_CONFIG.keys())
    weights = list(DEGREE_CONFIG.values())
    num_parents = random.choices(options, weights=weights)[0]
    
    if num_parents > len(allowed_vars): 
        num_parents = len(allowed_vars)

    inputs = random.sample(allowed_vars, k=num_parents)
    
    rule_str = inputs[0]
    for i in range(1, num_parents):
        op = random.choice(["&", "|", "^"])
        operand = inputs[i]
        if random.random() < 0.2:
            operand = f"~{operand}"
        rule_str = f"({rule_str} {op} {operand})"

    if random.random() < 0.15:
        rule_str = f"~({rule_str})"
        
    return rule_str

def generate_case(n, m, p, T, verbose=False):
    if verbose:
        print(f"--- Generating a BCN instance with output rules y=h(x) (N={n}, M={m}, P={p}, T={T}) ---")
    f_rules = {}
    x_and_u_vars = [f"x{k}" for k in range(n)] + [f"u{k}" for k in range(m)]
    for i in range(n):
        f_rules[i] = generate_logic_rule(x_and_u_vars)

    h_rules = {}
    x_only_vars = [f"x{k}" for k in range(n)]
    for k in range(p):
        h_rules[k] = generate_logic_rule(x_only_vars)

    x0 = [random.randint(0, 1) for _ in range(n)]
    target_flip_size = max(1, p // 2)
    ground_truth_flip_nodes = random.sample(range(n), target_flip_size)
    
    raw_min = int(math.sqrt(T) / 2) + 2
    raw_max = int(math.sqrt(T)) + 2
    
    min_flips = max(1, raw_min)
    max_flips = max(min_flips, raw_max)
    
    if verbose:
        print(f"[Configuration] Flip-set size: {target_flip_size}")
        print(f"[Configuration] Flip-count range per node: [{min_flips}, {max_flips}]")
        print(f"[Ground truth] Preset flipped nodes: {ground_truth_flip_nodes}")
    flip_schedule = {t: [] for t in range(T)}
    total_flips = 0
    
    for node in ground_truth_flip_nodes:
        count = random.randint(min_flips, max_flips)
        times = random.sample(range(T), count)
        for t in times:
            flip_schedule[t].append(node)
            total_flips += 1
            
    if verbose:
        print(f"[Ground truth] Total flip actions: {total_flips}")
    current_x = x0[:]
    targets = []
    u_seq = [[random.randint(0, 1) for _ in range(m)] for _ in range(T)]

    for t in range(T):
        next_x = [0] * n
        state_dict_f = {f"x{i}": current_x[i] for i in range(n)}
        for j in range(m): state_dict_f[f"u{j}"] = u_seq[t][j]
        for i in range(n):
            next_x[i] = evaluate_logic(f_rules[i], state_dict_f)
        if t in flip_schedule:
            for node in flip_schedule[t]:
                next_x[node] = 1 - next_x[node]
        
        current_x = next_x
        state_dict_h = {f"x{i}": current_x[i] for i in range(n)}
        for out_k in range(p):
            y_val = evaluate_logic(h_rules[out_k], state_dict_h)
            targets.append((out_k, t + 1, y_val))

    return {
        "n": n, "m": m, "p": p, "T": T,
        "f_rules": f_rules,      
        "h_rules": h_rules,      
        "x0": x0,
        "targets": targets,      
        "ground_truth_flip_set": ground_truth_flip_nodes,
        "debug_flip_schedule": flip_schedule
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Random BCN instance generator (y=h(x) format)")
    parser.add_argument("--n", type=int, default=30, help="Number of state nodes (default: 30)")
    parser.add_argument("--m", type=int, default=3, help="Number of control inputs (default: 3)")
    parser.add_argument("--p", type=int, default=6, help="Number of output nodes (default: 6)")
    parser.add_argument("--T", type=int, default=30, help="Tracking horizon (default: 30)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible experiments")
    parser.add_argument("-o", "--output", default="bcn_case_data_h_x.json", help="Output file name")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data = generate_case(args.n, args.m, args.p, args.T, verbose=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {args.output}")
