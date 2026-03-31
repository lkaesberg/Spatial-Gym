import os, re
from pathlib import Path
import statistics
from collections import defaultdict, Counter

LOG_DIR = Path("logfiles")
PATTERNS = {
    "terminated": re.compile(
        r"Episode\s+\d+\s+terminated\s+after\s+(?P<steps>\d+)\s+steps;\s+final\s+reward=(?P<reward>[-\d\.]+)\s*;\s*difficulty=(?P<diff>\d+)"
    ),
    "truncated": re.compile(
        r"Episode\s+\d+\s+truncated\s+after\s+(?P<steps>\d+)\s+steps;\s+final\s+reward=(?P<reward>[-\d\.]+)\s*;\s*difficulty=(?P<diff>\d+)"
    ),
    "ran_full": re.compile(
        r"Episode\s+\d+\s+ran\s+full\s+(?P<steps>\d+)\s+steps;\s+final\s+reward=(?P<reward>[-\d\.]+)\s*;\s+difficulty=(?P<diff>\d+)"
    ),
}

def parse_log(path: Path):
    result = {
        "puzzle": int(path.stem.replace("puzzle", "")),
        "status": None,
        "steps": None,
        "reward": None,
        "difficulty": None,
        "comp_tokens": []
    }
    
    content = path.read_text(encoding="utf-8")
    is_invalid_output = "aborted due to invalid output" in content
    
    for line in path.read_text(encoding="utf-8").splitlines():
        for status, pat in PATTERNS.items():
            m = pat.search(line)
            if m:
                final_status = "invalid_output" if is_invalid_output and status == "truncated" else status
                result.update({
                    "status": final_status,
                    "steps": int(m.group("steps")),
                    "reward": float(m.group("reward")),
                    "difficulty": int(m.group("diff")),
                })
        m2 = re.search(r"completion_tokens=(\d+)", line)
        if m2:
            result["comp_tokens"].append(int(m2.group(1)))
    return result

if __name__ == "__main__":
    all_results = []
    for f in sorted(LOG_DIR.glob("puzzle*.log"), key=lambda p: int(p.stem.replace("puzzle",""))):
        all_results.append(parse_log(f))

    for r in all_results:
        ct = r["comp_tokens"]
        if ct:
            r["ct_sum"] = sum(ct)                     
            r["ct_avg"] = statistics.mean(ct)
            r["ct_med"] = statistics.median(ct)
            r["ct_min"] = min(ct)
            r["ct_max"] = max(ct)
        else:
            r["ct_sum"] = r["ct_avg"] = r["ct_med"] = r["ct_min"] = r["ct_max"] = 0


    avgs = [r["ct_avg"] for r in all_results]
    meds = [r["ct_med"] for r in all_results]
    mins = [r["ct_min"] for r in all_results]
    maxs = [r["ct_max"] for r in all_results]
    
        
    overall = {
        "avg": statistics.mean(avgs),
        "med": statistics.median(meds),
        "min": min(mins),
        "max": max(maxs)
    }
    
    # print table
    header = ["puzzle", "status", "steps", "reward", "difficulty"]
    print("  ".join(h.ljust(10) for h in header))
    print("-" * 55)
    for r in all_results:
        print(
            f"{r['puzzle']:<10}{r['status'] or 'N/A':<10}"
            f"{(r['steps'] or '—'):<10}{(r['reward'] or '—'):<10}"
            f"{(r['difficulty'] or '—'):<10}"
        )
    

    statuses = [r["status"] for r in all_results]
    counts = Counter(statuses)
    steps = [r["steps"] for r in all_results if r["steps"] is not None]
    step_stats = {
        "avg": statistics.mean(steps),
        "med": statistics.median(steps),
        "min": min(steps),
        "max": max(steps)
    }
    sums = [r["ct_sum"] for r in all_results]
    sum_stats = {
        "avg": statistics.mean(sums),
        "med": statistics.median(sums),
        "min": min(sums),
        "max": max(sums)
    }
    
    total     = len(all_results)
    wins      = sum(1 for r in all_results if r.get("reward") == 1)
    fails     = sum(1 for r in all_results if r.get("reward") == -1)
    truncated = counts.get("truncated", 0)
    terminated_count = counts.get("terminated", 0)
    invalid_output_count = counts.get("invalid_output", 0) 
    win_pct  = wins  / total * 100 if total else 0.0
    fail_pct = fails / total * 100 if total else 0.0
    term_pct  = terminated_count / total * 100 if total else 0.0
    trunc_pct = truncated / total * 100 if total else 0.0
    invalid_pct = invalid_output_count / total * 100 if total else 0.0


    summary = (
        f"total puzzles: {total}\n"
        f"wins: {win_pct:.2f}% ({wins})\n"
        f"fails: {fail_pct:.2f}% ({fails})\n"
        f"terminated runs:      {term_pct:.2f}% ({terminated_count})\n"
        f"truncated runs:       {trunc_pct:.2f}% ({truncated})\n"
        f"invalid model output: {invalid_pct:.2f}% ({invalid_output_count})\n"
        f"average_completion_tokens_per_puzzle:\n"
        f"  avg={overall['avg']:.2f}, "
        f"med={overall['med']:.2f}, "
        f"min={overall['min']}, "
        f"max={overall['max']}\n"
        f"steps_per_puzzle:\n"
        f"  avg={step_stats['avg']:.2f}, "
        f"med={step_stats['med']:.2f}, "
        f"min={step_stats['min']}, "
        f"max={step_stats['max']}\n"
        f"total_completion_tokens_per_puzzle:\n"
        f"  avg={sum_stats['avg']:.2f}, "
        f"med={sum_stats['med']:.2f}, "
        f"min={sum_stats['min']}, "
        f"max={sum_stats['max']}\n"
    )

    with open("logs_results/summary.txt", "w", encoding="utf-8") as outf:
        outf.write(summary)
        
    by_diff: dict[int, list[dict]] = defaultdict(list)
    for rec in all_results:
        diff = rec.get("difficulty")
        if diff is None:
            continue
        by_diff[diff].append(rec)

    # write a per‐difficulty summary
    with open("logs_results/summary_by_difficulty.txt", "w", encoding="utf-8") as outf:
        for diff in sorted(by_diff):
            group = by_diff[diff]
            total     = len(group)
            statuses  = [r["status"] for r in group]
            counts    = Counter(statuses)
            wins      = sum(1 for r in group if r.get("reward") == 1)
            fails     = sum(1 for r in group if r.get("reward") == -1)
            terminated= counts.get("terminated", 0)
            truncated = counts.get("truncated", 0)
            invalid_output = counts.get("invalid_output", 0)
            win_d_pct  = wins  / total * 100 if total else 0.0
            fail_d_pct = fails / total * 100 if total else 0.0
            term_d_pct  = terminated / total * 100 if total else 0.0
            trunc_d_pct = truncated / total * 100 if total else 0.0
            invalid_d_pct = invalid_output / total * 100 if total else 0.0
            avgs_d = [r["ct_avg"] for r in group]
            meds_d = [r["ct_med"] for r in group]
            mins_d = [r["ct_min"] for r in group]
            maxs_d = [r["ct_max"] for r in group]
            stats_d = {
                "avg": statistics.mean(avgs_d),
                "med": statistics.median(meds_d),
                "min": min(mins_d),
                "max": max(maxs_d)
            }
            steps_d = [r["steps"] for r in group if r["steps"] is not None]
            sd = {
                "avg": statistics.mean(steps_d),
                "med": statistics.median(steps_d),
                "min": min(steps_d),
                "max": max(steps_d)
            }
            sums_d = [r["ct_sum"] for r in group]
            sum_stats_d = {
                "avg": statistics.mean(sums_d),
                "med": statistics.median(sums_d),
                "min": min(sums_d),
                "max": max(sums_d)
            }

            outf.write(f"Difficulty {diff}\n")
            outf.write(f"  total puzzles:        {total}\n")
            outf.write(f"  wins:       {win_d_pct:.2f}% ({wins})\n")
            outf.write(f"  fails:     {fail_d_pct:.2f}% ({fails})\n")
            outf.write(f"  terminated runs:     {term_d_pct:.2f}% ({terminated})\n")
            outf.write(f"  truncated runs:      {trunc_d_pct:.2f}% ({truncated})\n")
            outf.write(f"  invalid model output: {invalid_d_pct:.2f}% ({invalid_output})\n") # Add to per-difficulty summary
            outf.write(
                f"  average_completion_tokens_per_puzzle: "
                f"avg={stats_d['avg']:.2f}, "
                f"med={stats_d['med']:.2f}, "
                f"min={stats_d['min']}, "
                f"max={stats_d['max']}\n\n"
            )
            outf.write(
                f"  steps_per_puzzle: "
                f"avg={sd['avg']:.2f}, "
                f"med={sd['med']:.2f}, "
                f"min={sd['min']}, "
                f"max={sd['max']}\n\n"
            )
            outf.write(
                f"  total_completion_tokens_per_puzzle: "
                f"avg={sum_stats_d['avg']:.2f}, "
                f"med={sum_stats_d['med']:.2f}, "
                f"min={sum_stats_d['min']}, "
                f"max={sum_stats_d['max']}\n"
            )
                        
    crashed = [r["puzzle"] for r in all_results if r.get("status") is None]

    with open("logs_results/crashed_puzzles.txt", "w", encoding="utf-8") as f:
        f.write("Crashed puzzle indices (log file present but no result):\n")
        if crashed:
            f.write(", ".join(str(i) for i in crashed))
        else:
            f.write("None")
        f.write("\n")