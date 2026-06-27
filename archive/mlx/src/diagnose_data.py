"""Diagnose data quality and training bottlenecks."""
import json

data = json.load(open("data/gsm8k_train_parsed.json"))
print(f"Total samples: {len(data)}")

correct = 0
wrong = 0
errs = []

for i, d in enumerate(data):
    trace = d["trace"]
    target = d["target"]
    vv = {}
    for s in trace.get("steps", []):
        op = s.get("op", "const")
        a1 = s.get("arg1", 0)
        a2 = s.get("arg2", 0)

        def r(a):
            if isinstance(a, (int, float)):
                return float(a)
            if isinstance(a, str) and a in vv:
                return vv[a]
            try:
                return float(a)
            except:
                return 0.0

        v1, v2 = r(a1), r(a2)
        if op == "add": res = v1 + v2
        elif op == "sub": res = v1 - v2
        elif op == "mul": res = v1 * v2
        elif op == "div" and v2 != 0: res = v1 / v2
        else: res = v1
        rk = s.get("result", "")
        if rk:
            vv[rk] = res

    fv = trace.get("final_answer", "")
    comp = vv.get(fv, 0)
    if abs(int(round(comp)) - int(round(target))) == 0:
        correct += 1
    else:
        wrong += 1
        if len(errs) < 10:
            errs.append(f"  #{i}: computed={int(round(comp))} target={int(round(target))}")

print(f"Correct traces: {correct} ({100*correct/len(data):.1f}%)")
print(f"WRONG traces:   {wrong} ({100*wrong/len(data):.1f}%)")
print()
if errs:
    print("Sample wrong traces:")
    for e in errs:
        print(e)

# Answer distribution
from collections import Counter
targets = [int(d["target"]) for d in data]
c = Counter(targets)
print(f"\nUnique answers: {len(c)}")
print(f"1-digit: {sum(1 for t in targets if abs(t)<10)}")
print(f"2-digit: {sum(1 for t in targets if 10<=abs(t)<100)}")
print(f"3-digit: {sum(1 for t in targets if 100<=abs(t)<1000)}")
print(f"4+ digit: {sum(1 for t in targets if abs(t)>=1000)}")
