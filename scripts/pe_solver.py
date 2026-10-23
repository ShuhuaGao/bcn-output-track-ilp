import json
import itertools
import time
import argparse

class BoolInt(int):
    """Represent Boolean values with integer-compatible bitwise semantics."""
    def __new__(cls, value): return super().__new__(cls, 1 if value else 0)
    def __invert__(self): return BoolInt(1 - self)
    def __and__(self, other): return BoolInt(super().__and__(other))
    def __or__(self, other): return BoolInt(super().__or__(other))
    def __xor__(self, other): return BoolInt(super().__xor__(other))
    def __rand__(self, other): return BoolInt(other & self)
    def __ror__(self, other): return BoolInt(other | self)
    def __rxor__(self, other): return BoolInt(other ^ self)

class SetReachabilitySolver:
    def __init__(self, data):
        t_build_start = time.time()

        self.n = data['n']
        self.m = data['m']
        self.p = data['p']
        self.T = data['T']
        self.x0_int = int("".join(map(str, data['x0'])), 2)
        self.f_rules = {int(k): v for k, v in data['f_rules'].items()}
        self.h_rules = {int(k): v for k, v in data['h_rules'].items()}
        self.compiled_f_rules = {
            i: compile(self.f_rules.get(i, f"x{i}"), '<string>', 'eval')
            for i in range(self.n)
        }
        self.compiled_h_rules = {
            k: compile(self.h_rules.get(k, "0"), '<string>', 'eval')
            for k in range(self.p)
        }
        self.targets_by_time = {t: [] for t in range(1, self.T + 1)}
        for item in data['targets']:
            out_k, t_idx, val = item
            if t_idx <= self.T:
                self.targets_by_time[t_idx].append((out_k, val))
        self.dynamics_cache = {}
        self.h_cache = {}
        self.all_u_ints = list(range(2 ** self.m))

        self.construction_time = time.time() - t_build_start
        self.solver_time = 0.0
        self.total_time = 0.0

    def _int_to_bool_list(self, val, size):
        return [(val >> (size - 1 - i)) & 1 for i in range(size)]

    def _compute_and_cache_nominal(self, x_int, u_int):
        """Implement the  compute and cache nominal operation.
        
        Args:
        x_int: Input used by this operation.
        u_int: Input used by this operation.
        """
        x_vec = self._int_to_bool_list(x_int, self.n)
        u_vec = self._int_to_bool_list(u_int, self.m)

        env = {f"x{i}": BoolInt(x_vec[i]) for i in range(self.n)}
        for j in range(self.m):
            env[f"u{j}"] = BoolInt(u_vec[j])

        next_x_int = 0
        for i in range(self.n):
            try:
                val = int(eval(self.compiled_f_rules[i], {}, env))
            except Exception:
                val = 0

            if val:
                next_x_int |= (1 << (self.n - 1 - i))

        self.dynamics_cache[(x_int, u_int)] = next_x_int
        return next_x_int

    def _satisfies_targets(self, x_int, t):
        """Implement the  satisfies targets operation.
        
        Args:
        x_int: Input used by this operation.
        t: Input used by this operation.
        """
        targets = self.targets_by_time[t]
        if not targets:
            return True

        x_vec = None
        env = None

        for out_k, expected_val in targets:
            cache_key = (x_int, out_k)
            if cache_key in self.h_cache:
                actual_val = self.h_cache[cache_key]
            else:
                if env is None:
                    x_vec = self._int_to_bool_list(x_int, self.n)
                    env = {f"x{i}": BoolInt(x_vec[i]) for i in range(self.n)}
                try:
                    actual_val = int(eval(self.compiled_h_rules[out_k], {}, env))
                except Exception:
                    actual_val = 0
                self.h_cache[cache_key] = actual_val
            if actual_val != expected_val:
                return False

        return True

    def verify_reachability(self, candidate_flip_set):
        """Implement the verify reachability operation.
        
        Args:
        candidate_flip_set: Input used by this operation.
        """
        current_reachable_set = {self.x0_int}
        reachability_tree = {t: {} for t in range(self.T + 1)}
        valid_flip_masks = []
        for r in range(len(candidate_flip_set) + 1):
            for sub in itertools.combinations(candidate_flip_set, r):
                mask = 0
                for idx in sub:
                    mask |= (1 << (self.n - 1 - idx))
                valid_flip_masks.append((mask, list(sub)))

        cache = self.dynamics_cache

        for t in range(self.T):
            next_reachable_set = {}

            for x_int in current_reachable_set:
                for u_int in self.all_u_ints:
                    x_nom_int = cache.get((x_int, u_int))
                    if x_nom_int is None:
                        x_nom_int = self._compute_and_cache_nominal(x_int, u_int)
                    for mask, action in valid_flip_masks:
                        x_final_int = x_nom_int ^ mask
                        if self._satisfies_targets(x_final_int, t + 1):
                            if x_final_int not in next_reachable_set:
                                next_reachable_set[x_final_int] = (x_int, u_int, action)

            if not next_reachable_set:
                return False, None

            reachability_tree[t+1] = next_reachable_set
            current_reachable_set = set(next_reachable_set.keys())
        final_state_int = next(iter(current_reachable_set))
        trajectory = []
        curr = final_state_int

        for t in range(self.T, 0, -1):
            prev, u_int, action = reachability_tree[t][curr]
            u_vec = self._int_to_bool_list(u_int, self.m)
            trajectory.insert(0, (t-1, u_vec, action))
            curr = prev

        return True, trajectory

def find_minimal_flip_set_by_reachability(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None
    solver = SetReachabilitySolver(data)
    n = data['n']

    print(f"--- Algorithm: Set-based forward reachability analysis (y=h(x)) ---")
    print(f"    Network size N={n}, horizon T={data['T']}")
    start_time = time.time()

    for k in range(n + 1):
        print(f"\n>>> Testing flip-set size k={k} ...")

        combinations = list(itertools.combinations(range(n), k))
        total_combs = len(combinations)
        print(f"    Combinations to test: {total_combs} node combinations")

        for idx, subset in enumerate(combinations):
            subset = list(subset)
            if idx % 10 == 0:
                print(f"    Analyzing combination {idx}/{total_combs}...".ljust(40), end='\r')

            is_feasible, trace = solver.verify_reachability(subset)

            if is_feasible:
                end_time = time.time()
                solver.solver_time = end_time - start_time
                solver.total_time = solver.construction_time + solver.solver_time

                print(f"\n    Combination {subset} passed analysis. [SUCCESS]".ljust(40))
                print("\n" + "="*60)
                print(f"Solution found")
                print(f"Construction Time (build/preprocessing): {solver.construction_time:.4f} s")
                print(f"Solver Time       (search):    {solver.solver_time:.4f} s")
                print(f"Total Time        (total):    {solver.total_time:.4f} s")
                print(f"Minimum flip set (Beta): {subset}")
                print(f"Cache sizes: dynamics {len(solver.dynamics_cache)} entries, observations {len(solver.h_cache)} entries")
                print("="*60)
                print(f"{'T':<3} | {'u(t)':<12} | {'Flip Action (v)'}")
                print("-" * 45)

                total_instant_flips = 0
                for item in trace:
                    t_idx, u_val, v_val = item
                    total_instant_flips += len(v_val)
                    print(f"{t_idx:<3} | {str(u_val):<12} | {str(v_val)}")
                print("-" * 45)
                print(f"Total instantaneous flips (Total Instant Flips): {total_instant_flips}")

                return {
                    'construction_time': solver.construction_time,
                    'solver_time': solver.solver_time,
                    'total_time': solver.total_time,
                    'beta': subset,
                }

    end_time = time.time()
    solver.solver_time = end_time - start_time
    solver.total_time = solver.construction_time + solver.solver_time

    print("\nNo feasible solution found (infeasible).")
    print(f"Construction Time (build/preprocessing): {solver.construction_time:.4f} s")
    print(f"Solver Time       (search):    {solver.solver_time:.4f} s")
    print(f"Total Time        (total):    {solver.total_time:.4f} s")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PE (pruned enumeration) solver for BCNs")
    parser.add_argument("filename", nargs="?", default="bcn_case_data_h_x.json",
                        help="BCN data file (JSON; default: bcn_case_data_h_x.json; generate it with bcn_generator.py first)")
    args = parser.parse_args()
    find_minimal_flip_set_by_reachability(args.filename)
