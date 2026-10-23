# -*- coding: utf-8 -*-
"""Experimental code for cpsat solver."""

import json
import time
import re
import os
from ortools.sat.python import cp_model

class LogicParser:
    """Implement the LogicParser component."""
    def __init__(self):
        self.precedence = {'~': 4, '&': 3, '^': 2, '|': 1, '(': 0}

    def tokenize(self, expr):
        """Convert a Boolean expression string into tokens.
        
        Args:
        expr: Input used by this operation.
        """
        expr = expr.replace(' ', '')
        tokens = re.findall(r'[xu]\d+|[&|^~()]', expr)
        return tokens

    def to_postfix(self, tokens):
        """Convert infix expression tokens to reverse Polish notation.
        
        Args:
        tokens: Input used by this operation.
        """
        output_queue = []
        operator_stack = []

        for token in tokens:
            if token.startswith('x') or token.startswith('u'):
                output_queue.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()
            else:
                while (operator_stack and operator_stack[-1] != '(' and
                       self.precedence[operator_stack[-1]] >= self.precedence[token]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)

        while operator_stack:
            output_queue.append(operator_stack.pop())

        return output_queue

class StateFlippedBCN_CpSat:
    def __init__(self, data, max_time_seconds=None, num_workers=None, log_search=False, verbose=True):
        self.n = data['n']
        self.m = data['m']
        self.p = data['p']
        self.T = data['T']
        self.f_rules = {int(k): v for k, v in data['f_rules'].items()}
        self.h_rules = {int(k): v for k, v in data['h_rules'].items()}

        self.x0 = data['x0']
        self.targets = data['targets']
        self.truth_set = data.get('ground_truth_flip_set', [])

        self.model = None
        self.solver = None
        self.parser = LogicParser()
        self.verbose = verbose
        self.max_time_seconds = max_time_seconds
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 8)
        self.log_search = log_search
        self.construction_time = 0.0
        self.solver_time = 0.0
        self.total_time = 0.0
        self._x = None
        self._u = None
        self._f = None
        self._z = None

    def _add_gate_constraints(self, op_type, y, inputs, prefix):
        """Implement the  add gate constraints operation.
        
        Args:
        op_type: Input used by this operation.
        y: Input used by this operation.
        inputs: Input used by this operation.
        prefix: Input used by this operation.
        """
        model = self.model

        if op_type == '~':  # NOT: y = 1 - a
            model.add(y + inputs[0] == 1)

        elif op_type == '&':
            model.add_bool_and(inputs).only_enforce_if(y)
            model.add_bool_or([v.Not() for v in inputs]).only_enforce_if(y.Not())

        elif op_type == '|':
            model.add_bool_or(inputs).only_enforce_if(y)
            model.add_bool_and([v.Not() for v in inputs]).only_enforce_if(y.Not())

        elif op_type == '^':
            a, b = inputs[0], inputs[1]
            tmp = self.model.new_bool_var(f"xor_aux_{prefix}")
            model.add_multiplication_equality(tmp, [a, b])  # tmp = a & b
            model.add(y == a + b - 2 * tmp)

    def _build_constraints_from_postfix(self, postfix_tokens, var_map, prefix):
        """Implement the  build constraints from postfix operation.
        
        Args:
        postfix_tokens: Input used by this operation.
        var_map: Input used by this operation.
        prefix: Input used by this operation.
        """
        stack = []
        step_counter = 0

        for token in postfix_tokens:
            if token.startswith('x') or token.startswith('u'):
                stack.append(var_map[token])
            else:
                res_var = self.model.new_bool_var(f"w_{prefix}_{step_counter}")
                step_counter += 1

                if token == '~':
                    op1 = stack.pop()
                    operands = [op1]
                else:
                    op2 = stack.pop()
                    op1 = stack.pop()
                    operands = [op1, op2]

                self._add_gate_constraints(token, res_var, operands, f"{prefix}_{step_counter}")
                stack.append(res_var)

        return stack[0]

    def solve(self):
        if self.verbose:
            print(f"Building the CP-SAT model (T={self.T}, format: y=h(x))...")
        t_total_start = time.time()
        t_build_start = time.time()

        self.model = cp_model.CpModel()
        x = {(i, t): self.model.new_bool_var(f"x_{i}_{t}")
             for i in range(self.n) for t in range(self.T + 1)}
        u = {(j, t): self.model.new_bool_var(f"u_{j}_{t}")
             for j in range(self.m) for t in range(self.T)}
        f = {(i, t): self.model.new_bool_var(f"f_{i}_{t}")
             for i in range(self.n) for t in range(self.T)}
        z = {i: self.model.new_bool_var(f"z_{i}") for i in range(self.n)}

        self._x, self._u, self._f, self._z = x, u, f, z
        for i in range(self.n):
            self.model.add(x[i, 0] == int(self.x0[i]))
        for t in range(self.T):
            var_map = {f"x{i}": x[i, t] for i in range(self.n)}
            for j in range(self.m):
                var_map[f"u{j}"] = u[j, t]

            for i in range(self.n):
                rule_str = self.f_rules.get(i, f"x{i}")

                tokens = self.parser.tokenize(rule_str)
                postfix = self.parser.to_postfix(tokens)
                xn_val = self._build_constraints_from_postfix(postfix, var_map, f"f_node{i}_t{t}")
                self._add_gate_constraints('^', x[i, t + 1], [xn_val, f[i, t]], f"FlipLogic_{i}_{t}")
                self.model.add(z[i] >= f[i, t])
        for item in self.targets:
            out_k, time_idx, val = item

            var_map_h = {f"x{i}": x[i, time_idx] for i in range(self.n)}
            rule_str_h = self.h_rules[out_k]

            tokens_h = self.parser.tokenize(rule_str_h)
            postfix_h = self.parser.to_postfix(tokens_h)

            y_val_var = self._build_constraints_from_postfix(postfix_h, var_map_h, f"h_node{out_k}_t{time_idx}")
            self.model.add(y_val_var == int(val))
        self.model.minimize(sum(z[i] for i in range(self.n)))

        t_build_end = time.time()
        self.construction_time = t_build_end - t_build_start
        self.solver = cp_model.CpSolver()
        self.solver.parameters.num_search_workers = self.num_workers
        if self.max_time_seconds is not None:
            self.solver.parameters.max_time_in_seconds = float(self.max_time_seconds)
        if self.log_search:
            self.solver.parameters.log_search_progress = True

        status = self.solver.solve(self.model)
        self.solver_time = self.solver.wall_time

        t_total_end = time.time()
        self.total_time = t_total_end - t_total_start
        result = {
            'status': self.solver.status_name(status),
            'construction_time': self.construction_time,
            'solver_time': self.solver_time,
            'total_time': self.total_time,
            'obj_val': None,
            'success': False,
        }

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            obj_val = int(round(self.solver.objective_value))
            flipped_indices = sorted([i for i in range(self.n) if self.solver.value(z[i]) > 0])

            result['obj_val'] = obj_val
            result['success'] = (status == cp_model.OPTIMAL)

            if self.verbose:
                print("\n" + "=" * 60)
                print(f"[Solution found] (cp_sat, status: {result['status']})")
                print(f"Construction Time (model construction): {self.construction_time:.4f}s")
                print(f"Solver Time       (solve time): {self.solver_time:.4f}s")
                print(f"Total Time        (total): {self.total_time:.4f}s")
                print(f"Minimum number of flipped nodes (Beta Size): {obj_val}")
                print(f"Flip set returned by the solver: {flipped_indices}")
                print(f"Preset ground-truth flip set: {sorted(self.truth_set)}")
                print("=" * 60)

                print(f"{'T':<3} | {'u(t)':<10} | {'Flip Action (v_i=1)'}")
                print("-" * 45)
                for t in range(self.T):
                    u_vals = [self.solver.value(u[j, t]) for j in range(self.m)]
                    active_flips = [i for i in range(self.n) if self.solver.value(f[i, t]) > 0]
                    print(f"{t:<3} | {str(u_vals):<10} | {str(active_flips)}")
                print("-" * 45)
        else:
            if self.verbose:
                print(f"The model is infeasible or no solution was found (status: {result['status']})")
                print(f"Construction Time (model construction): {self.construction_time:.4f}s")
                print(f"Solver Time       (solve time): {self.solver_time:.4f}s")
                print(f"Total Time        (total): {self.total_time:.4f}s")

        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CP-SAT solver for the proposed BCN ILP method")
    parser.add_argument("filename", nargs="?", default="bcn_case_data_h_x.json",
                        help="BCN data file (JSON, f_rules/h_rules format; default: bcn_case_data_h_x.json")
    parser.add_argument("--max-time", type=float, default=None, help="CP-SAT time limit (seconds)")
    args = parser.parse_args()

    try:
        with open(args.filename, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        solver = StateFlippedBCN_CpSat(data, max_time_seconds=args.max_time)
        solver.solve()
    except FileNotFoundError:
        print(f"Error: file not found {args.filename}，Run the generator first.")
