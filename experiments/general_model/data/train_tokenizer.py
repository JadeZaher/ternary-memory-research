"""
experiments/general_model/data/train_tokenizer.py: train the general-model 16k byte-level BPE.

Thin CLI over prepare_general_corpus.stage_tokenizer so the tokenizer can be rebuilt without
re-running the corpus build. Writes data/general/tokenizer.json and data/general/tokenizer_report.json
(bytes/token per source, new tokenizer vs GPT-2). Why 16k byte-level: see AGENTS.md in this directory.

Run:  python experiments/general_model/data/train_tokenizer.py --tokenizer-sample-mb 200
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from experiments.general_model.data.prepare_general_corpus import (  # noqa: E402
    build_argparser,
    resolve_paths,
    stage_tokenizer,
)


def main() -> None:
    ap = build_argparser()
    ap.set_defaults(stage="tokenizer")
    args = ap.parse_args()
    resolve_paths(args)
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))
    out = stage_tokenizer(args)
    print(json.dumps({"vocab_size": out["vocab_size"], "eos_id": out["eos_id"],
                      "tokenizer": args.tokenizer_path}, indent=2))


if __name__ == "__main__":
    main()
