import os, subprocess, time

# 轮询：等四方法 × 三轴全部走到 10 seeds，且对应 run_teachers 进程全部退出后，
# 生成 extend_done.flag。避免原 extend_seeds.sh 里 wait 报错导致 flag 永不落盘。
methods_files = [
    ("results/teachers_erm_epcon.json", ["erm", "epcon"]),
    ("results/teachers_supcon.json", ["supcon"]),
    ("results/teachers_gpu1.json", ["groupdro"]),
]
AXES = ["loc", "ori", "user"]

def all_done():
    import json
    for f, ms in methods_files:
        if not os.path.exists(f):
            return False
        try:
            d = json.load(open(f))
        except Exception:
            return False
        for m in ms:
            if m not in d:
                return False
            for ax in AXES:
                if ax not in d[m]:
                    return False
                if len(d[m][ax].get("acc", {})) < 10:
                    return False
    # 还要确保没有 run_teachers 进程还在写（避免读到半截状态误判）
    return True

def any_trainer_alive():
    r = subprocess.run(["pgrep", "-f", "run_teachers.py"],
                        capture_output=True, text=True)
    return r.returncode == 0  # pgrep 返回 0 = 找到了

print("watcher: polling until 4 methods x 3 axes reach 10 seeds AND trainers exit")
while True:
    done = all_done()
    alive = any_trainer_alive()
    if done and not alive:
        open("logs/extend_done.flag", "w").write("done")
        print("extend_done.flag written")
        break
    time.sleep(60)