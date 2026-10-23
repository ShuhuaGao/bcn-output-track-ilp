# -*- coding: utf-8 -*-
"""Experimental code for q learning solver."""

import json
import itertools
import random
import time
import argparse

import numpy as np


class BoolInt(int):
    def __new__(cls, value): return super().__new__(cls, 1 if value else 0)
    def __invert__(self): return BoolInt(1 - self)
    def __and__(self, other): return BoolInt(super().__and__(other))
    def __or__(self, other): return BoolInt(super().__or__(other))
    def __xor__(self, other): return BoolInt(super().__xor__(other))
    def __rand__(self, other): return BoolInt(other & self)
    def __ror__(self, other): return BoolInt(other | self)
    def __rxor__(self, other): return BoolInt(other ^ self)


class BCN_QLearning_Solver:
    def __init__(self, data):
        t_build_start = time.time()

        self.n = data['n']
        self.m = data['m']
        self.p = data['p']
        self.T = data['T']
        self.f_rules = {int(k): v for k, v in data['f_rules'].items()}
        self.h_rules = {int(k): v for k, v in data['h_rules'].items()}

        self.x0 = data['x0']
        self.compiled_f_rules = [
            compile(self.f_rules.get(i, f"x{i}"), '<string>', 'eval')
            for i in range(self.n)
        ]
        self.compiled_h_rules = [
            compile(self.h_rules.get(k, "0"), '<string>', 'eval')
            for k in range(self.p)
        ]
        self.targets_by_time = {t: [] for t in range(1, self.T + 1)}
        for item in data['targets']:
            out_k, t_idx, val = item
            if t_idx <= self.T:
                self.targets_by_time[t_idx].append((out_k, val))
        self.h_cache = {}

        self.construction_time = time.time() - t_build_start
        print(f"Building the STP state-transition matrix (Size: 2^{self.m} × 2^{self.n})...")
        t_stp_start = time.time()
        self.stp_matrix = self.build_stp_matrix()
        self.stp_matrix_time = time.time() - t_stp_start
        print(f"STP STP table construction completed in: {self.stp_matrix_time:.4f} s。")
        self.solver_time = 0.0
        self.total_time = 0.0

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def build_stp_matrix(self):
        """Implement the build stp matrix operation."""
        dim_x = 1 << self.n
        dim_u = 1 << self.m
        table = np.zeros((dim_u, dim_x), dtype=np.int32)
        X_arr = np.arange(dim_x, dtype=np.int32)

        for u in range(dim_u):
            env = {}
            for j in range(self.m):
                env[f"u{j}"] = bool((u >> (self.m - 1 - j)) & 1)
            for i in range(self.n):
                env[f"x{i}"] = ((X_arr >> (self.n - 1 - i)) & 1).astype(bool)

            next_x_arr = np.zeros(dim_x, dtype=np.int32)

            for i in range(self.n):
                rule = self.f_rules[i]
                safe_rule = rule.replace(' and ', ' & ').replace(' or ', ' | ').replace('not ', '~')

                val_arr = eval(safe_rule, {"__builtins__": None, "np": np}, env)

                if isinstance(val_arr, bool):
                    val_arr = np.full(dim_x, val_arr, dtype=bool)

                next_x_arr |= (val_arr.astype(np.int32) << (self.n - 1 - i))

            table[u, :] = next_x_arr

        return table

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def _bool_to_int(self, vec):
        res = 0
        for bit in vec:
            res = (res << 1) | bit
        return res

    def _int_to_bool(self, val, size):
        return [(val >> i) & 1 for i in range(size - 1, -1, -1)]

    def compute_next_state(self, x_vec, u_vec, flip_action):
        """Implement the compute next state operation.
        
        Args:
        x_vec: Input used by this operation.
        u_vec: Input used by this operation.
        flip_action: Input used by this operation.
        """
        x_int = self._bool_to_int(x_vec)
        u_int = self._bool_to_int(u_vec)
        next_int = int(self.stp_matrix[u_int, x_int])
        for idx in flip_action:
            next_int ^= (1 << (self.n - 1 - idx))

        return self._int_to_bool(next_int, self.n)

    def check_output_constraint(self, x_vec, x_int, t):
        """Implement the check output constraint operation.
        
        Args:
        x_vec: Input used by this operation.
        x_int: Input used by this operation.
        t: Input used by this operation.
        """
        targets = self.targets_by_time.get(t, [])
        if not targets:
            return True

        env = None
        for out_k, expected_val in targets:
            cache_key = (x_int, out_k)

            if cache_key in self.h_cache:
                actual_val = self.h_cache[cache_key]
            else:
                if env is None:
                    env = {f"x{i}": BoolInt(x_vec[i]) for i in range(self.n)}
                try:
                    actual_val = int(eval(self.compiled_h_rules[out_k], {}, env))
                except Exception:
                    actual_val = 0
                self.h_cache[cache_key] = actual_val

            if actual_val != expected_val:
                return False
        return True

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def find_minimum_flip_set(self):
        print("\n[Phase 1] Running reachability analysis to find the minimum flip set (β_min)...")
        for k in range(self.n + 1):
            combinations = list(itertools.combinations(range(self.n), k))
            for subset in combinations:
                subset = list(subset)
                if self._verify_reachability(subset):
                    print(f"  => Minimum flip set found β_min: {subset}")
                    return subset
        return None

    def _verify_reachability(self, candidate_set):
        current_set = {self._bool_to_int(self.x0)}
        all_u = [self._int_to_bool(u, self.m) for u in range(2 ** self.m)]
        valid_flips = []
        for r in range(len(candidate_set) + 1):
            for sub in itertools.combinations(candidate_set, r):
                valid_flips.append(list(sub))

        for t in range(self.T):
            next_set = set()
            for x_int in current_set:
                x_vec = self._int_to_bool(x_int, self.n)
                for u_vec in all_u:
                    for action in valid_flips:
                        x_next_vec = self.compute_next_state(x_vec, u_vec, action)
                        x_next_int = self._bool_to_int(x_next_vec)

                        if self.check_output_constraint(x_next_vec, x_next_int, t + 1):
                            next_set.add(x_next_int)

            if not next_set:
                return False
            current_set = next_set
        return True

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def train_q_learning(self, beta_min, episodes=5000, alpha=0.5, gamma=0.9):
        print(f"\n[Phase 2] Initializing the Q-learning agent...")

        action_space = []
        for u in range(2 ** self.m):
            u_vec = tuple(self._int_to_bool(u, self.m))
            for r in range(len(beta_min) + 1):
                for sub in itertools.combinations(beta_min, r):
                    action_space.append((u_vec, tuple(sub)))

        print(f"  => Action-space size: |A| = {len(action_space)}")

        q_table = {}

        def get_q(t, x_int, a):
            if (t, x_int) not in q_table:
                q_table[(t, x_int)] = {act: 0.0 for act in action_space}
            return q_table[(t, x_int)][a]

        def set_q(t, x_int, a, val):
            if (t, x_int) not in q_table:
                q_table[(t, x_int)] = {act: 0.0 for act in action_space}
            q_table[(t, x_int)][a] = val

        print(f"  => Starting training (Episodes={episodes})...")
        start_time = time.time()

        epsilon_start = 1.0
        epsilon_min = 0.05
        epsilon_decay = 0.995
        epsilon = epsilon_start

        for ep in range(episodes):
            x_vec = list(self.x0)
            x_int = self._bool_to_int(x_vec)

            for t in range(self.T):
                state_key = (t, x_int)

                if random.random() < epsilon:
                    action = random.choice(action_space)
                else:
                    if state_key not in q_table:
                        action = random.choice(action_space)
                    else:
                        action = max(q_table[state_key], key=q_table[state_key].get)

                u_vec, flip_action = action
                next_x_vec = self.compute_next_state(x_vec, u_vec, flip_action)
                next_x_int = self._bool_to_int(next_x_vec)

                is_valid = self.check_output_constraint(next_x_vec, next_x_int, t + 1)

                if is_valid:
                    reward = 100.0 if t == self.T - 1 else 10.0
                    done = False
                else:
                    reward = -50.0
                    done = True

                current_q = get_q(t, x_int, action)
                if done or t == self.T - 1:
                    max_next_q = 0.0
                else:
                    if (t + 1, next_x_int) not in q_table:
                        max_next_q = 0.0
                    else:
                        max_next_q = max(q_table[(t + 1, next_x_int)].values())

                new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
                set_q(t, x_int, action, new_q)

                if done:
                    break

                x_vec = next_x_vec
                x_int = next_x_int

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if (ep + 1) % 1000 == 0:
                print(f"     [Episode {ep+1}/{episodes}] Epsilon: {epsilon:.3f}", end='\r')

        print(f"\n  => Training completed in: {time.time() - start_time:.2f} s。")
        return q_table, action_space

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def extract_optimal_trajectory(self, q_table, action_space):
        print("\n[Phase 3] Extracting the best joint control sequence from the Q-table...")
        x_vec = list(self.x0)
        x_int = self._bool_to_int(x_vec)
        trajectory = []

        for t in range(self.T):
            state_key = (t, x_int)
            if state_key not in q_table:
                print(f"  [!] Could not extract a trajectory: the Q-table has no reliable state at t={t}. The number of training episodes may be insufficient.")
                return None

            best_action = max(q_table[state_key], key=q_table[state_key].get)
            u_vec, flip_action = best_action

            trajectory.append((t, list(u_vec), list(flip_action)))

            x_vec = self.compute_next_state(x_vec, u_vec, flip_action)
            x_int = self._bool_to_int(x_vec)

            if not self.check_output_constraint(x_vec, x_int, t + 1):
                print(f"  [!] The extracted trajectory at t={t} does not satisfy the constraints; training may not have converged.")
                return None

        return trajectory

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def report_times(self):
        self.total_time = self.construction_time + self.stp_matrix_time + self.solver_time
        print(f"Construction Time (rule compilation/preprocessing): {self.construction_time:.4f} s")
        print(f"STP Matrix Time   (table construction):        {self.stp_matrix_time:.4f} s")
        print(f"Solver Time       (search and training):       {self.solver_time:.4f} s")
        print(f"Total Time        (total):        {self.total_time:.4f} s")


def main():
    parser = argparse.ArgumentParser(description="STP + Q-learning solver for BCNs")
    parser.add_argument("filename", nargs="?", default="bcn_case_data_h_x.json",
                        help="BCN data file (JSON; default: bcn_case_data_h_x.json; generate it with bcn_generator.py first)")
    parser.add_argument("--episodes", type=int, default=500000,
                        help="Number of Q-learning episodes; larger networks may require more episodes to converge")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible experiments")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        with open(args.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found {args.filename}。Run the generator first (python bcn_generator.py)。")
        return
    solver = BCN_QLearning_Solver(data)

    # Solver Time: Phase 1 + Phase 2 + Phase 3
    t0 = time.time()
    beta_min = solver.find_minimum_flip_set()

    if beta_min is not None:
        q_table, actions = solver.train_q_learning(beta_min, episodes=args.episodes, alpha=0.5, gamma=0.95)
        trace = solver.extract_optimal_trajectory(q_table, actions)
        solver.solver_time = time.time() - t0

        print("\n" + "=" * 60)
        if trace:
            print("Q-learning found a solution.")
            solver.report_times()
            print(f"Minimum flip set (β_min): {beta_min}")
            print(f"Number of cached output evaluations (h_cache): {len(solver.h_cache)}")
            print("-" * 50)
            print(f"{'t':<3} | {'Control u(t)':<15} | {'Flip Action α(t)'}")
            print("-" * 50)
            for t_idx, u_val, v_val in trace:
                print(f"{t_idx:<3} | {str(u_val):<15} | {str(v_val)}")
            print("=" * 60)
        else:
            print("Q-learning did not converge to a feasible trajectory. Try increasing the number of episodes.")
            solver.report_times()
    else:
        solver.solver_time = time.time() - t0
        print("The network has no feasible solution within the given horizon T.")
        solver.report_times()


if __name__ == "__main__":
    main()
