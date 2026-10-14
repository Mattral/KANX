# Contributing to kanx 🚀

First — thanks for considering a contribution! `kanx` is built to be
**easy to fork**, **easy to extend**, and **boring to maintain**. PRs of any
size are welcome, from typo fixes to new backends.

---

## TL;DR

```bash
git clone https://github.com/Mattral/KANX
cd KANX
pip install -e ".[dev,api,torch,onnx,docs]"
bash scripts/test.sh         # must stay green
```

Open a draft PR early. We'd rather discuss with code in front of us!

---

## Project values (read this before sending a PR)

1. **One canonical surface per concept.** If there's already a `train()`,
   don't add `fit_easy_v2()`. Improve the existing one.
2. **Fail loud at the boundary.** Validate inputs in public functions; raise
   `ValueError` / `HTTPException` with a precise message.
3. **Tiny core dependency surface.** `tensorflow + numpy + pyyaml` only. New
   deps go behind optional extras (`api`, `torch`, `onnx`, `docs`).
4. **Maths gets a test.** New numerical contracts → new test in
   `tests/test_layers.py` or `tests/test_properties.py`.
5. **Two backends, one semantics.** Anything you add to `kanx.X` should
   either:
    - have a parallel implementation in `kanx.torch.X`, **or**
    - be explicitly documented as backend-specific.

---

## How to find something to work on

- 🔖 [**Good first issues**](https://github.com/Mattral/KANX/labels/good%20first%20issue) — small, well-scoped tasks
- 🗺️ [`roadmap.md`](roadmap.md) — P0 / P1 / P2 backlog
- 🐛 [Open bugs](https://github.com/Mattral/KANX/issues)
- 💬 [Discussions](https://github.com/Mattral/KANX/discussions) — design questions

### High-leverage ideas (currently unowned)

- PyTorch subprocess benchmark wrapper for GPU runners — next
- Mixed precision and XLA on the spline einsum — next
- JAX backend (third parallel surface) — P1

---

## Local setup

```bash
# 1. fork + clone + cd
# 2. install
pip install -e ".[dev,api,torch,onnx,docs]"

# 3. run tests (must stay green BEFORE you start)
bash scripts/test.sh

# 4. start the API locally
uvicorn api.app:app --reload --port 8000

# 5. build docs locally
mkdocs serve
```

## Pull-request checklist

- [ ] One logical change per PR (mixing refactor + feature → split it)
- [ ] `bash scripts/test.sh` is green
- [ ] New behaviour has a new test
- [ ] `ruff check src tests api` is clean
- [ ] `CHANGELOG.md` updated under **`[Unreleased]`**
- [ ] `roadmap.md` updated if you shipped a roadmap item
- [ ] Public API change → update `README.md` + relevant doc in `docs/`
- [ ] PR description explains *why*, not just *what*

## Coding conventions

- **Imports** sorted with `ruff format` (PEP 8 / isort-compatible).
- **Type hints** on public functions; internal helpers are optional.
- **Docstrings** in Google style; include a short example for new public APIs.
- **No emoji** in source files (README and docs are fine).
- **No `print`** in library code — use `kanx.utils.get_logger()`.

## Reporting bugs

Open an issue with:

1. `kanx.__version__`, Python version, TF / PyTorch versions
2. Minimum reproducible snippet (≤ 20 lines)
3. Expected vs actual behaviour
4. Full stacktrace if it crashed

## Security disclosures

Please **don't** open a public issue for security bugs. Email the maintainer
directly (address in the project metadata) with details and a PoC.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE) — the same license as `kanx`.
