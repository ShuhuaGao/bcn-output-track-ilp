# -*- coding: utf-8 -*-
"""Experimental code for batch given networks."""

import json
import numpy as np
from cpsat_solver import StateFlippedBCN_CpSat
try:
    import resource
    def _peak_rss_mb():
        """Implement the  peak rss mb operation."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    def _peak_rss_mb():
        return None
FILES = [
    "data/n_10_t_10.json",
    "data/n_15_t_15.json",
    "data/n_15_t_20.json",
    "data/n_20_t_20.json",
    "data/n_25_t_25.json",
    "data/n_30_t_30.json",
]


def run_given_networks(files=FILES, k=5, output_file="given_networks_results.txt"):
    """Implement the run given networks operation.
    
    Args:
    files: Input used by this operation.
    k: Input used by this operation.
    output_file: Input used by this operation.
    """
    with open(output_file, "w", encoding="utf-8") as fp:
        def log(msg):
            """Write a message to the console and the experiment log.
            
            Args:
            msg: Input used by this operation.
            """
            print(msg)
            fp.write(msg + "\n")

        log(f"Starting batch solve for the supplied networks (solver: CP-SAT); networks: {len(files)} networks, repetitions per network: {k} times")
        log("=" * 100)

        summary_data = []

        for fn in files:
            try:
                with open(fn, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                log(f"\n>>> Skipping {fn}: file does not exist")
                continue

            n_val, t_val = data['n'], data['T']
            truth = sorted(data.get('ground_truth_flip_set', []))

            log(f"\n>>> Network: {fn} (N={n_val}, M={data['m']}, P={data['p']}, T={t_val})")
            log("-" * 100)
            log(f"{'Iter':<5} | {'Status':<10} | {'Build (s)':<10} | {'Solve (s)':<10} | {'Total (s)':<10} | {'PeakMem(MB)':<12} | {'ObjVal':<10} | {'Truth Size'}")

            build_times, solve_times, total_times, mem_peaks = [], [], [], []
            success_count = 0
            obj_vals = []

            for i in range(k):
                solver = StateFlippedBCN_CpSat(data, verbose=False)
                res = solver.solve()
                mem_peak = _peak_rss_mb()

                if res['success']:
                    success_count += 1
                    build_times.append(res['construction_time'])
                    solve_times.append(res['solver_time'])
                    total_times.append(res['total_time'])
                    if mem_peak is not None:
                        mem_peaks.append(mem_peak)
                    obj_vals.append(res['obj_val'])
                    status_str = "OPTIMAL"
                    obj_val_str = str(res['obj_val'])
                else:
                    status_str = "FAIL"
                    obj_val_str = "N/A"

                mem_str = f"{mem_peak:<12.1f}" if mem_peak is not None else f"{'N/A':<12}"
                log(f"{i+1:<5} | {status_str:<10} | {res['construction_time']:<10.4f} | {res['solver_time']:<10.4f} | {res['total_time']:<10.4f} | {mem_str} | {obj_val_str:<10} | {len(truth)}")

            if total_times:
                avg_b, avg_s, avg_t = np.mean(build_times), np.mean(solve_times), np.mean(total_times)
                max_t, min_t = np.max(total_times), np.min(total_times)
                avg_m = float(np.mean(mem_peaks)) if mem_peaks else None
                max_m = float(np.max(mem_peaks)) if mem_peaks else None
                success_rate = success_count / k
                obj_consistent = len(set(obj_vals)) == 1
                obj_show = obj_vals[0] if obj_consistent else obj_vals
                summary_data.append((fn, n_val, t_val, success_rate, avg_b, avg_s, avg_t, max_t, min_t, avg_m, max_m, obj_show, len(truth)))
                mem_part = f", average peak memory: {avg_m:.1f}MB, process peak memory: {max_m:.1f}MB" if avg_m is not None else ""
                log(f"--> [Summary] success rate: {success_rate*100:.1f}%, ObjVal: {obj_show}"
                    f"{'' if obj_consistent else ' (results differ between runs!)'}, "
                    f"mean construction: {avg_b:.4f}s, mean solve: {avg_s:.4f}s, mean total time: {avg_t:.4f}s{mem_part}")
            else:
                summary_data.append((fn, n_val, t_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, "N/A", len(truth)))
                log(f"--> [Summary] all instances failed (Infeasible/Timeout)")

        log("\n" + "=" * 140)
        log("【Global experiment summary (Global Summary)】")
        log(f"{'File':<18} | {'N':<4} | {'T':<4} | {'Succ%':<7} | {'Obj':<5} | {'Truth':<6} | "
            f"{'Avg Build':<10} | {'Avg Solve':<10} | {'Avg Total':<10} | {'Max Total':<10} | {'Min Total':<10} | "
            f"{'Avg PeakMB':<11} | {'Max PeakMB':<11}")
        log("-" * 140)

        for item in summary_data:
            fn, n_val, t_val, sr, avg_b, avg_s, avg_t, max_t, min_t, avg_m, max_m, obj_show, truth_size = item
            avg_m_str = f"{avg_m:>11.1f}" if avg_m is not None else f"{'N/A':>11}"
            max_m_str = f"{max_m:>11.1f}" if max_m is not None else f"{'N/A':>11}"
            log(f"{fn:<18} | {n_val:<4} | {t_val:<4} | {sr*100:>5.1f}% | {str(obj_show):<5} | {truth_size:<6} | "
                f"{avg_b:>10.4f} | {avg_s:>10.4f} | {avg_t:>10.4f} | {max_t:>10.4f} | {min_t:>10.4f} | "
                f"{avg_m_str} | {max_m_str}")

        log("=" * 140)
        final_peak = _peak_rss_mb()
        if final_peak is not None:
            log(f"\nProcess peak memory for the entire run (ru_maxrss): {final_peak:.1f} MB")

    print(f"\n>>> Batch solve complete. Results saved to: {output_file}")


if __name__ == "__main__":
    try:
        import numpy
    except ImportError:
        print("Error: NumPy is required. Install it with: uv add numpy")
        exit()
    run_given_networks(files=FILES, k=5, output_file="given_networks_results.txt")
