# -*- coding: utf-8 -*-
"""Experimental code for batch scale sweep."""

import numpy as np
import bcn_generator
from cpsat_solver import StateFlippedBCN_CpSat
try:
    import resource
    def _peak_rss_mb():
        """Implement the  peak rss mb operation."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    def _peak_rss_mb():
        return None

def run_range_experiments(r, s, step=5, k=10, output_file="range_results.txt"):
    """Run CP-SAT experiments over a range of network sizes.
    
    Args:
    r: Input used by this operation.
    s: Input used by this operation.
    step: Input used by this operation.
    k: Input used by this operation.
    output_file: Input used by this operation.
    """

    with open(output_file, "w", encoding="utf-8") as f:
        def log(msg):
            """Write a message to the console and the experiment log.
            
            Args:
            msg: Input used by this operation.
            """
            print(msg)
            f.write(msg + "\n")

        log(f"Starting scale-sweep experiment (solver: CP-SAT): N from {r} to {s}, step {step}")
        log(f"Repetitions per network size N: {k} times")
        log("=" * 70)
        summary_data = []

        for n_val in range(r, s + 1, step):
            m_val = max(1, n_val // 10)
            p_val = max(1, n_val // 5)
            t_val = n_val

            log(f"\n>>> Testing network size: N={n_val} (settings: M={m_val}, P={p_val}, T={t_val})")
            log("-" * 65)
            log(f"{'Iter':<5} | {'Status':<10} | {'Build (s)':<10} | {'Solve (s)':<10} | {'Total (s)':<10} | {'PeakMem(MB)':<12} | {'ObjVal':<10} | {'Truth Size'}")

            build_times = []
            solve_times = []
            results = []
            mem_peaks = []
            success_count = 0

            for i in range(k):
                case_data = bcn_generator.generate_case(n_val, m_val, p_val, t_val, verbose=False)
                truth_size = len(case_data['ground_truth_flip_set'])
                solver = StateFlippedBCN_CpSat(case_data, verbose=False)
                res = solver.solve()
                mem_peak = _peak_rss_mb()
                if res['success']:
                    success_count += 1
                    build_times.append(res['construction_time'])
                    solve_times.append(res['solver_time'])
                    results.append(res['total_time'])
                    if mem_peak is not None:
                        mem_peaks.append(mem_peak)
                    status_str = "OPTIMAL"
                    obj_val_str = str(res['obj_val'])
                else:
                    status_str = "FAIL"
                    obj_val_str = "N/A"

                mem_str = f"{mem_peak:<12.1f}" if mem_peak is not None else f"{'N/A':<12}"
                log(f"{i+1:<5} | {status_str:<10} | {res['construction_time']:<10.4f} | {res['solver_time']:<10.4f} | {res['total_time']:<10.4f} | {mem_str} | {obj_val_str:<10} | {truth_size}")
            if results:
                avg_b = np.mean(build_times)
                avg_s = np.mean(solve_times)
                avg_t = np.mean(results)
                max_t = np.max(results)
                min_t = np.min(results)
                avg_m = float(np.mean(mem_peaks)) if mem_peaks else None
                max_m = float(np.max(mem_peaks)) if mem_peaks else None
                success_rate = success_count / k
                summary_data.append((n_val, success_rate, avg_b, avg_s, avg_t, max_t, min_t, avg_m, max_m))
                mem_part = f", average peak memory: {avg_m:.1f}MB, process peak memory: {max_m:.1f}MB" if avg_m is not None else ""
                log(f"--> [N={n_val} Summary] success rate: {success_rate*100:.1f}%, mean construction: {avg_b:.4f}s, mean solve: {avg_s:.4f}s, mean total time: {avg_t:.4f}s{mem_part}")
            else:
                summary_data.append((n_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None))
                log(f"--> [N={n_val} Summary] all instances failed (Infeasible/Timeout)")

        log("\n" + "=" * 130)
        log("【Global experiment summary (Global Summary)】")
        log(f"{'N Size':<8} | {'Success Rate':<13} | {'Avg Build (s)':<14} | {'Avg Solve (s)':<14} | {'Avg Total (s)':<14} | {'Max Total (s)':<14} | {'Min Total (s)':<14} | {'Avg Peak (MB)':<14} | {'Max Peak (MB)':<14}")
        log("-" * 130)

        for item in summary_data:
            n_val, sr, avg_b, avg_s, avg_t, max_t, min_t, avg_m, max_m = item
            avg_m_str = f"{avg_m:>14.1f}" if avg_m is not None else f"{'N/A':>14}"
            max_m_str = f"{max_m:>14.1f}" if max_m is not None else f"{'N/A':>14}"
            log(f"N={n_val:<6} | {sr*100:>11.1f}% | {avg_b:>14.4f} | {avg_s:>14.4f} | {avg_t:>14.4f} | {max_t:>14.4f} | {min_t:>14.4f} | {avg_m_str} | {max_m_str}")

        log("=" * 130)
        final_peak = _peak_rss_mb()
        if final_peak is not None:
            log(f"\nProcess peak memory for the entire run (ru_maxrss): {final_peak:.1f} MB")

    print(f"\n>>> Experiment complete. Detailed logs and summary saved to: {output_file}")

if __name__ == "__main__":
    try:
        import numpy
    except ImportError:
        print("Error: NumPy is required for statistics. Install it with: uv add numpy")
        exit()
    run_range_experiments(r=10, s=70, step=5, k=25, output_file="sweep_results.txt")
