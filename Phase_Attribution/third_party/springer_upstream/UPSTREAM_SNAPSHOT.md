Verbatim snapshot of https://github.com/EchoStatements/Springer-Segmentation-Python
at commit 276b9cc (fetched 2026-09-07). Only the library modules are kept
(no tiny_test data, no full_training_script.py, no matlab_preprocessing.py).

Do not edit these files. They exist so `segmenter_eval/compare_upstream_vs_local.py`
can train and evaluate the unmodified original segmenter side by side with
`segmenter/`. The modules use flat imports (`from utils import ...`), so the
comparison script puts this directory at the front of sys.path before
importing them; they never collide with the `segmenter.*` package.
