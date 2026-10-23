# -*- coding: utf-8 -*-
"""Experimental code for stp solver."""

import json
import itertools
import time
import argparse

import numpy as np


class Fast_STP_Solver:
    def __init__(self, data):
        self.n = data['n']
        self.m = data['m']
        self.p = data['p']
        self.T = data['T']
        self.f_rules = {int(k): v for k, v in data['f_rules'].items()}
        self.h_rules = {int(k): v for k, v in data['h_rules'].items()}
        self.x0_int = int("".join(map(str, data['x0'])), 2)
        self.targets_by_time = {t: [] for t in range(1, self.T + 1)}
        for item in data['targets']:
            out_k, t_idx, val = item
            if t_idx <= self.T:
                self.targets_by_time[t_idx].append((int(out_k), int(val)))
        print(f"Building the STP state-transition lookup table (Size: 2^{self.m} × 2^{self.n})...")
        t0 = time.time()
        self.transition_table = self.build_transition_table_vectorized()
        self.output_table = self.build_output_table_vectorized()
        self.construction_time = time.time() - t0
        print(f"STP table construction completed in: {self.construction_time:.4f} s。")
        self.solver_time = 0.0
        self.total_time = 0.0

    @staticmethod
    def _eval_rule_vectorized(rule, env):
        """Implement the  eval rule vectorized operation.
        
        Args:
        rule: Input used by this operation.
        env: Input used by this operation.
        """
        safe_rule = rule.replace(' and ', ' & ').replace(' or ', ' | ').replace('not ', '~')
        val_arr = eval(safe_rule, {"__builtins__": None, "np": np}, env)
        if not isinstance(val_arr, np.ndarray):
            dim = len(env["x0"])
            val_arr = np.full(dim, bool(val_arr), dtype=bool)
        return val_arr

    def _state_bit_env(self, X_arr):
        """Implement the  state bit env operation.
        
        Args:
        X_arr: Input used by this operation.
        """
        return {f"x{i}": ((X_arr >> (self.n - 1 - i)) & 1).astype(bool)
                for i in range(self.n)}

    def build_transition_table_vectorized(self):
        """Implement the build transition table vectorized operation."""
        dim_x = 1 << self.n
        dim_u = 1 << self.m
        table = np.zeros((dim_u, dim_x), dtype=np.int32)

        X_arr = np.arange(dim_x, dtype=np.int32)

        for u in range(dim_u):
            env = self._state_bit_env(X_arr)
            for j in range(self.m):
                env[f"u{j}"] = bool((u >> (self.m - 1 - j)) & 1)

            next_x_arr = np.zeros(dim_x, dtype=np.int32)
            for i in range(self.n):
                val_arr = self._eval_rule_vectorized(self.f_rules[i], env)
                next_x_arr |= (val_arr.astype(np.int32) << (self.n - 1 - i))

            table[u, :] = next_x_arr

        return table

    def build_output_table_vectorized(self):
        """Implement the build output table vectorized operation."""
        dim_x = 1 << self.n
        obs = np.zeros((self.p, dim_x), dtype=np.int32)

        X_arr = np.arange(dim_x, dtype=np.int32)
        env = self._state_bit_env(X_arr)

        for k in range(self.p):
            val_arr = self._eval_rule_vectorized(self.h_rules[k], env)
            obs[k, :] = val_arr.astype(np.int32)

        return obs

    def solve(self):
        print(f"Starting the STP solver (N={self.n})...")
        t_solve_start = time.time()

        result = None
        for k in range(self.n + 1):
            print(f">>> Checking flip-set size k={k} ...")
            combinations = list(itertools.combinations(range(self.n), k))
            for subset in combinations:
                if self.check_controllability(subset):
                    result = list(subset)
                    break
            if result is not None:
                break

        self.solver_time = time.time() - t_solve_start
        self.total_time = self.construction_time + self.solver_time
        return result

    def check_controllability(self, flip_nodes):
        bit_indices = [self.n - 1 - i for i in flip_nodes]
        flip_masks = []
        for r in range(len(bit_indices) + 1):
            for sub in itertools.combinations(bit_indices, r):
                mask = 0
                for bit in sub:
                    mask |= (1 << bit)
                flip_masks.append(mask)
        flip_masks_arr = np.array(flip_masks, dtype=np.int32)

        current_states = {self.x0_int}
        dim_u = 1 << self.m

        for t in range(self.T):
            next_states = set()
            targets = self.targets_by_time.get(t + 1, [])
            curr_states_arr = np.fromiter(current_states, dtype=np.int32)

            for u in range(dim_u):
                nom_next_arr = self.transition_table[u, curr_states_arr]
                flipped_matrix = nom_next_arr[:, None] ^ flip_masks_arr

                if targets:
                    valid = np.ones(flipped_matrix.shape, dtype=bool)
                    for out_k, val in targets:
                        valid &= (self.output_table[out_k, flipped_matrix] == val)
                    next_states.update(flipped_matrix[valid].ravel())
                else:
                    next_states.update(flipped_matrix.ravel())

            if not next_states:
                return False

            current_states = next_states

        return True


def main():
    parser = argparse.ArgumentParser(description="Vectorized STP solver")
    parser.add_argument("filename", nargs="?", default="data/n_20_t_20.json",
                        help="BCN data file (JSON, f_rules/h_rules format; default: data/n_20_t_20.json")
    args = parser.parse_args()

    try:
        with open(args.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {args.filename}. Check the file name.")
        return

    solver = Fast_STP_Solver(data)
    res = solver.solve()

    print(f"\nConstruction Time (table construction): {solver.construction_time:.4f} s")
    print(f"Solver Time       (search): {solver.solver_time:.4f} s")
    print(f"Total Time        (total): {solver.total_time:.4f} s (vectorized STP)")
    if res is not None:
        print(f"Solution found. Minimum flip set: {res}")
    else:
        print("No feasible solution found")


if __name__ == "__main__":
    main()
