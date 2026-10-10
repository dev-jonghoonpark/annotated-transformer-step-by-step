"""단계의 demo.py를 실행하고 출력(output.txt)과 차트(charts/)를 그 단계 폴더에 저장한다.

    uv run python tools/run_step.py 03        # 3단계만
    uv run python tools/run_step.py all       # 전부 (recorded 단계는 건너뜀)

데이터 캐시(.data)와 vocab.pt는 저장소 루트의 .work/에 둔다.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work"


def run(step_dir):
    if (step_dir / "RECORDED.md").exists():
        print(f"skip {step_dir.name} (recorded)")
        return
    charts = step_dir / "charts"
    shutil.rmtree(charts, ignore_errors=True)
    WORK.mkdir(exist_ok=True)
    env = dict(os.environ, CHART_DIR=str(charts), PYTHONPATH=str(ROOT / "tools"), PYTHONHASHSEED="0")
    print(f"run {step_dir.name} ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(step_dir / "demo.py")],
        cwd=WORK, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"{step_dir.name} 실패\n{proc.stdout}\n{proc.stderr}")
    (step_dir / "output.txt").write_text(proc.stdout)


if __name__ == "__main__":
    steps = sorted(p for p in (ROOT / "steps").iterdir() if p.is_dir())
    arg = sys.argv[1]
    for s in steps:
        if arg == "all" or s.name.startswith(arg.zfill(2)):
            run(s)
