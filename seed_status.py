import json
for f, methods in [
    ("results/teachers_erm_epcon.json", ["erm", "epcon"]),
    ("results/teachers_supcon.json", ["supcon"]),
    ("results/teachers_gpu1.json", ["groupdro"]),
]:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, "ERR", e); continue
    for m in methods:
        if m not in d: 
            print(f, m, "MISSING"); continue
        for ax in ["loc", "ori", "user"]:
            n = len(d[m][ax]["acc"])
            print("%-30s %-8s %-5s %d/10 seeds" % (f, m, ax, n))