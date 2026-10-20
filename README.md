# ageval-qa

Datasets for [ageval](https://github.com/ZJU-REAL/ageval), the agent evaluation runtime. Each directory is one `ageval.dataset/1` package (`ageval.yaml` plus `tasks/`) and is run with the `ageval` CLI. Documentation: <https://zju-real.github.io/ageval/en/docs/>.

Two question-answering datasets. Each directory is its own dataset root. `dataset_id` is unchanged.

| Directory | dataset_id | Tasks |
| --- | --- | --- |
| `hle/` | `official/hle` | 573 (HLE text-only gold) |
| `widesearch/` | `official/widesearch` | 200 |

```bash
ageval lock hle --task 668825f80a642802bdfeadfa
ageval run widesearch --task ws_en_001
```

Put secrets in each dataset's `.env`, copied from that directory's `env.example`. `.env` and `.ageval/` stay out of version control.

HLE's canary forbids training on the questions. It does not forbid publishing them. Per-dataset license and gaps are in each directory's `README.md` and `NOTICE`.
