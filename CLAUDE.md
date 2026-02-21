# CLAUDE.md

Hi Claude! Welcome to the LargeCrimsonCanine repository. We are glad to have you here and appreciate your contribution to this project.

LargeCrimsonCanine is a high-performance geometric algebra library for Python, built on a Rust backend. The goal is to be the best geometric algebra library available — fast, correct, and genuinely pleasant to use.

Please take a moment to orient yourself before diving in.

## Start Here

Before writing any code, read these documents in order:

1. **ARCHITECTURE.md** — the design philosophy, technical decisions, and the principles that guide everything in this repo. This is the most important document. If you are unsure why something is built the way it is, the answer is probably here.

2. **CONTRIBUTING.md** — code comment conventions, citation policy, and the PR process. These are non-negotiable. Please follow them precisely.

## Core Principles to Keep in Mind

**Meet users where they are.** LCC does not ask users to change their workflow. Every API decision should minimize friction for people coming from NumPy, PyTorch, or existing geometric algebra libraries.

**Error messages teach.** Every error is an opportunity. Prefer specific, actionable, educational messages over terse ones.

**Comments explain why, not what.** If the code is clear, the comment should explain the reasoning behind it, not restate what it does. See CONTRIBUTING.md for examples.

**All citations are marked [VERIFY].** Do not add references from memory. Mathematical claims require independently verified sources. Mark anything unverified.

**The Rust backend is invisible to Python users.** If a Python user has to think about Rust to use LCC, something has gone wrong.

## What We Are Building

LCC implements the Clifford algebra Cl(p,q,r) framework. The core mathematical objects are multivectors. The fundamental operation is the geometric product. Everything else builds from there.

Current status and known limitations are documented in CHANGELOG.md.

## Branch and Worktree Workflow

main is protected. nothing goes directly to main. all work happens on branches via pull requests.

### Branch naming

- `feat/description` — new operations or capabilities
- `fix/description` — bug fixes
- `docs/description` — documentation only
- `perf/description` — performance improvements
- `chore/description` — infrastructure, tooling, CI

### Standard branch workflow

```bash
# Create a branch for your work
git checkout -b feat/inner-product

# Do your work, commit regularly with clear messages
git commit -m "Add inner product for grade-1 multivectors"

# Push and open a PR against main
git push origin feat/inner-product
```

### Worktree workflow (for parallel workstreams)

If you are running as a Claude Code agent and need to work in parallel with other agents, use worktrees to avoid interference:

```bash
# Create a worktree for isolated work
claude --worktree feat/inner-product

# Or manually
git worktree add .worktrees/inner-product -b feat/inner-product
```

Each worktree gets its own branch and working directory. Agents in separate worktrees cannot interfere with each other. This is the recommended approach for large batched changes — split the work across multiple agents, each in their own worktree, each opening their own PR.

### Commit messages

- Present tense: "Add inner product" not "Added inner product"
- Specific: name the operation or file
- No vague messages like "fix bug" or "update code"

See CONTRIBUTING.md for the full PR checklist.

## Getting Started

```bash
# Install Rust (stable)
curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin (builds Rust + Python together)
pip install maturin pytest

# Build and install locally
maturin develop

# Run tests
cargo test        # Rust tests
pytest tests/     # Python tests
```

## Before Opening a PR

- Tests pass: `cargo test` and `pytest tests/`
- No clippy warnings: `cargo clippy -- -D warnings`
- Formatting clean: `cargo fmt --check`
- CHANGELOG.md updated under [Unreleased]
- All new citations marked [VERIFY]

Thank you for being here. Good work matters and we are glad you are doing it.

