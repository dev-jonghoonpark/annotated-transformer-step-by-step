"""데모에서 altair 차트를 웹 페이지에 넣기 위한 도우미.

차트를 CHART_DIR(기본: ./charts)에 Vega-Lite JSON으로 저장하고, 출력에 [[chart:이름]] 표시를 남긴다.
site/build.py가 이 표시 자리에 차트를 끼워 넣는다.
"""

import os
from pathlib import Path


def show(chart, name):
    out = Path(os.environ.get("CHART_DIR", "charts"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(chart.to_json())
    print(f"[[chart:{name}]]", flush=True)
