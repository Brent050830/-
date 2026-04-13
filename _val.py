import json, glob, os
print(f"{'File':30} | Eval  | Stress | Track OK")
for s in ['eco', 'normal', 'sport']:
    for f in sorted(glob.glob(f'results/{s}/seed_4*/metrics.json')):
        d = json.load(open(f))
        try:
            ev = d["eval_metrics"]["saving_total_pct"]
            st = d["stress_metrics"]["saving_total_pct"]
            tok = d["stress_metrics"]["tracking_ok"]
            print(f"{f:30} | {ev:>5.2f}% | {st:>6.2f}% | {tok}")
        except:
            print(f"{f:30} | Error parsing")
