# Contributing to OpenDrive-XAI

First off, thanks for taking the time to contribute! 🚗🛠️

This project follows the [Open Source Guides](https://opensource.guide/) best practices. The short version:

1. **Fork** the repo and create your branch from `main`.
2. **Code** your improvement with clear commits and comments.
3. **Lint & test** locally: `make lint` / `pytest`.
4. **Submit a pull request**—fill out the PR template and link any relevant issues.

## Code style

* Python: Black (line length = 88) + isort + flake8.
* Shell: `shellcheck` passes.
* C++/CUDA: `clang-format` (LLVM style).

Run all linters with:

```bash
./scripts/dev/lint_all.sh
```

## Commit messages

Follow Conventional Commits (`feat:`, `fix:`, `docs:` …). Example:

```
feat(planning): add MPC fallback controller
```

## Pull-request checklist

- [ ] Tests added/updated
- [ ] Docs updated (`docs/` or README snippets)
- [ ] CI passes (`GitHub Actions` status green)
- [ ] Linked to an issue (or explain the rationale)

## Communication

Join the project discussion under GitHub Issues or our Discord (link in README). Be respectful and inclusive—this project adheres to the Code of Conduct. 