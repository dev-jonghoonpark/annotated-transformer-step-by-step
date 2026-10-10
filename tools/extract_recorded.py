"""17·18단계 출력은 GPU 학습이 필요해서 빌드 때 다시 돌리지 않는다.
annotated-transformer-modern의 AnnotatedTransformer.ipynb(RTX 3070에서 끝까지 실행한 기록)에서 꺼내 온다.

    uv run python tools/extract_recorded.py ../annotated-transformer-modern/AnnotatedTransformer.ipynb
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell_by_source(nb, prefix):
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "".join(c["source"]).lstrip().startswith(prefix):
            return c
    raise KeyError(prefix)


def stream(cell):
    return "".join("".join(o["text"]) for o in cell["outputs"] if o["output_type"] == "stream")


def vega_spec(cell):
    html = next("".join(o["data"]["text/html"]) for o in cell["outputs"] if "data" in o)
    start = html.index("})(", html.index("function(spec, embedOpt)")) + 3
    spec, _ = json.JSONDecoder().raw_decode(html[start:])
    return spec


def main(nb_path):
    nb = json.loads(Path(nb_path).read_text())

    d = ROOT / "steps" / "17-training-multi30k"
    train_log = stream(cell_by_source(nb, "def train_distributed_model"))
    example = stream(cell_by_source(nb, "def viz_encoder_self"))
    (d / "output.txt").write_text(
        train_log.rstrip() + "\n\n" + example[example.index("Preparing Data"):].rstrip() + "\n"
    )

    d = ROOT / "steps" / "18-attention-visualization"
    (d / "charts").mkdir(exist_ok=True)
    parts = []
    for fn, name, title in [
        ("def viz_encoder_self", "encoder_self", "인코더 셀프 어텐션"),
        ("def viz_decoder_self", "decoder_self", "디코더 셀프 어텐션"),
        ("def viz_decoder_src", "decoder_src", "디코더 소스 어텐션"),
    ]:
        cell = cell_by_source(nb, fn)
        (d / "charts" / f"{name}.json").write_text(json.dumps(vega_spec(cell)))
        s = stream(cell)
        ex = s[s.index("Source Text"):].strip()
        parts.append(f"── {title} ──\n\n{ex}\n\n[[chart:{name}]]\n")
    (d / "output.txt").write_text("\n".join(parts))


if __name__ == "__main__":
    main(sys.argv[1])
