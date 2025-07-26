import json

filename = "synthetic_en_hi.jsonl"
valid, total, empty, identical = 0, 0, 0, 0

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        src, tgt = d["source"].strip(), d["target"].strip()
        if not tgt:
            empty += 1
        elif src == tgt:
            identical += 1
        else:
            valid += 1
        total += 1

print(f"Total: {total} | Valid: {valid} | Empty: {empty} | Identical: {identical}")
print(f"  {(valid/total)*100:.1f}% are usable synthetic translations")
