# -*- coding: utf-8 -*-
"""Experimental code for example 6node case."""

import json


def create_6node_case():
    n = 6
    m = 1
    p = 3
    T = 6
    rules = {
        0: "u0 & (~x5)",
        1: "~x0",
        2: "(~x0) & (x4 | x2)",
        3: "x0 & (~x5)",
        4: "x3 | (~x2)",
        5: "x4 & ((~x5) | (~x1))"
    }
    h_rules = {0: "x0", 1: "x1", 2: "x5"}
    x0 = [1] * n
    raw_fixed_y = [
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 1), (0, 4, 0), (0, 5, 1),
        (1, 0, 0), (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 0), (1, 5, 1),
        (2, 0, 1), (2, 1, 1), (2, 2, 0), (2, 3, 1), (2, 4, 0), (2, 5, 0)
    ]
    targets = [(out_k, t_raw + 1, val) for out_k, t_raw, val in raw_fixed_y]
    data = {
        "n": n,
        "m": m,
        "p": p,
        "T": T,
        "f_rules": rules,
        "h_rules": h_rules,
        "x0": x0,
        "targets": targets,
        "ground_truth_flip_set": []
    }
    filename = "data/6_nodes_data.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated six-node example data: {filename}")


if __name__ == "__main__":
    create_6node_case()
