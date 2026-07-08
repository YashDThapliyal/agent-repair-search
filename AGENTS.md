# AGENTS.md

## 0. Purpose of this file

This repository is designed to be built and maintained primarily by coding agents. Treat this file as the authoritative repository-level engineering contract.

Read this file completely before making changes. For any non-trivial task, inspect the repository state, relevant source files, tests, and `PLANS.md` before editing. Do not guess at APIs, dependency behavior, or experiment results.

The goal is not to produce a large platform. The goal is to produce the smallest rigorous open-source research prototype that can test one concrete hypothesis about improving tool-using agents.

---

# 1. Mission

Build a local-first open-source prototype for **optimizer-backed agent repair search**.

The system starts from:

1. a small tool-using agent,
2. a known/diagnosed failure mode,
3. editable agent artifacts,
4. an evaluation dataset split into optimization-train, optimization-validation,
   held-out, and regression sets.

It compares three experimental arms:

1. **Original agent** — no repair.
2. **Single-shot repair** — one LLM-generated best-effort patch from the diagnosis.
3. **Optimizer-backed repair search** — multiple candidate repairs are generated, evaluated, improved, and selected using an optimization loop; the finalist is then evaluated on held-out and regression sets.

The primary research question is:

> Given a diagnosed agent failure and issue-specific evals, does optimizer-backed candidate repair search outperform a single-shot LLM-proposed repair on held-out tool-calling behavior without introducing regressions?

The repository must make this question easy to reproduce and inspect.

---

# 2. Product thesis

The prototype is not an observability product and not a generic eval platform.

It demonstrates one narrow loop:

```text
Diagnosed failure
      ↓
Editable agent artifacts
      ↓
Generate candidate repairs
      ↓
Evaluate on optimization-train split
      ↓
Iterate/search over candidates
      ↓
Select finalist with optimization-validation
      ↓
Evaluate on held-out split
      ↓
Run regression suite
      ↓
Return validated patch + evidence
```

The compelling output is not a dashboard. It is a **measured, reproducible repair result**:

```text
Original agent             X%
Single-shot repair         Y%
Optimizer-backed repair    Z%

Held-out delta             +N pp
Regression gate            PASS / FAIL
Cost delta                 ...
Latency delta              ...
```

and a human-readable patch showing what changed.

---

# 3. Hard scope boundaries

## 3.1 Required for v0.1

Implement exactly the following core scope:

- Python project managed with `uv`.
- One synthetic but realistic customer-support tool-calling agent.
- Anthropic API as the model backend.
- Five tools:
  - `search_customer`
  - `lookup_subscription`
  - `cancel_subscription`
  - `issue_refund`
  - `escalate_to_human`
- Two editable artifact classes:
  - global system prompt,
  - tool descriptions.
- Four eval splits:
  - optimization-train,
  - optimization-validation,
  - held-out,
  - regression.
- Deterministic tool-call evaluators.
- Three experimental arms:
  - original,
  - single-shot repair,
  - optimizer-backed repair search.
- Reproducible experiment output.
- A validated textual diff/patch for the winning repair.
- Unit tests and at least one end-to-end smoke test.
- Clear README with one-command reproduction.

## 3.2 Explicit non-goals for v0.1

Do not build any of the following unless required for basic correctness:

- web frontend,
- React app,
- FastAPI service,
- hosted API,
- database,
- Postgres,
- Redis,
- background queue,
- Kubernetes,
- Docker dependency for normal use,
- tracing backend,
- vector database,
- production authentication,
- multi-tenant platform,
- live MCP server,
- browser automation,
- automatic deployment,
- generic support for every agent framework,
- autonomous production edits,
- broad "self-improving agents" platform abstractions.

Prefer a small, inspectable CLI and filesystem artifacts.

---

# 4. Repository positioning and README constraints

## 4.1 Public positioning

The repository should be positioned as:

> A reproducible experiment for optimizer-backed repair search in tool-using agents.

Recommended one-line description:

> Compare original, single-shot, and optimizer-backed repairs for tool-using agents with held-out and regression evaluation.

## 4.2 README hard rule

**Do not mention LangChain or LangSmith anywhere in `README.md`.**

This is a strict requirement.

Also:

- Do not market the repository as an extension of any commercial product.
- Do not claim a company or project lacks a feature.
- Do not include competitor comparisons in the README.
- Do not mention private internship work, internal company systems, internal prompts, internal traces, or confidential architectures.
- Do not claim benchmark gains unless they were actually produced by committed experiment artifacts.
- Do not fabricate results.

The README should stand alone as a clean open-source research prototype.

---

# 5. Anthropic model backend

Use the official Anthropic Python SDK.

## 5.1 Environment variables

Required:

```text
ANTHROPIC_API_KEY
ANTHROPIC_TASK_MODEL
ANTHROPIC_REPAIR_MODEL

# Optional backward-compatible shared model:
ANTHROPIC_MODEL
```

Rules:

- Do not commit API keys.
- Do not hardcode a model identifier in source code.
- Require a resolved task model and repair model at runtime, either through the
  role-specific variables or the backward-compatible shared `ANTHROPIC_MODEL`.
- `.env.example` may contain placeholders only.
- Fail with a clear actionable error if required credentials/config are missing.
- Centralize all Anthropic calls behind one small model-client abstraction.
- Capture token usage and latency when the SDK response exposes them.
- Add retry handling for transient failures with bounded exponential backoff.
- Never silently retry forever.
- Keep temperature and generation settings explicit and configurable.

## 5.2 Model roles

Use separate resolved model roles:

- `ANTHROPIC_TASK_MODEL` for every customer-support agent rollout in every arm.
- `ANTHROPIC_REPAIR_MODEL` for single-shot repair generation, GEPA candidate
  proposals, and repair reflection/planning calls.

`ANTHROPIC_MODEL` remains accepted only as a shared backward-compatible fallback
when a role-specific variable is absent. Do not silently invent a model ID.

---

# 6. Demo agent specification

Build one deterministic tool-selection evaluation environment around a synthetic customer-support domain.

## 6.1 Tool definitions

Represent tools as JSON-compatible schemas with at least:

```json
{
  "name": "cancel_subscription",
  "description": "...",
  "input_schema": {
    "type": "object",
    "properties": {}
  }
}
```

The actual runtime may use Anthropic tool-use structures, but internal schemas should be easy to inspect and mutate.

## 6.2 Intended failure mode

Create a deliberate but plausible ambiguity between cancellation and refund behavior.

Example failure cluster:

> Cancellation requests that mention money, billing, charges, or prior payments are incorrectly routed to `issue_refund` instead of `cancel_subscription`.

The failure must not be a toy typo. It should be caused by realistic interaction between:

- global instructions,
- overlapping tool descriptions,
- user phrasing.

The initial artifacts should be imperfect enough to create measurable failures but not so broken that any trivial edit solves all cases.

## 6.3 Agent output

Normalize each run into a typed structure similar to:

```python
@dataclass(frozen=True)
class AgentResult:
    final_answer: str | None
    tool_name: str | None
    tool_args: dict[str, object]
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    raw_response: object | None
```

Do not let evaluation code depend directly on raw SDK response shapes.

---

# 7. Dataset design

## 7.1 Required splits

Commit three datasets:

```text
evals/optimize_train.jsonl
evals/optimize_val.jsonl
evals/heldout.jsonl
evals/regression.jsonl
```

Suggested total size for the initial public experiment:

- optimization-train: roughly 30–45 cases,
- optimization-validation: roughly 10–20 cases,
- held-out: 20–30 cases,
- regression: 20–30 cases.

A somewhat smaller dataset is acceptable for the first smoke run, but the final committed benchmark should include enough lexical and semantic variation to make held-out evaluation meaningful.

## 7.2 Eval case schema

Each JSONL row should include fields like:

```json
{
  "id": "cancel-017",
  "input": "Stop my Pro plan after this billing period. I was charged yesterday.",
  "expected_tool": "cancel_subscription",
  "expected_args": {
    "when": "end_of_billing_cycle"
  },
  "category": "cancellation",
  "failure_cluster": "billing_language_routes_to_refund",
  "notes": "Held-out wording combines cancellation intent with charge language."
}
```

Not every row needs `failure_cluster` or `notes`, but IDs and expected behavior must be explicit.

## 7.3 Leakage prevention

This is critical.

- The optimizer may use only `optimize_train` for repair feedback and
  `optimize_val` for internal candidate selection.
- Do not expose held-out expected labels to candidate generation or selection.
- The held-out split is used only after candidate selection.
- The regression split must cover unrelated valid behaviors, especially legitimate refund requests.
- Never choose the final candidate based on held-out performance.
- Record split hashes in result artifacts so experiments are auditable.

---

# 8. Metrics and evaluator contract

Implement deterministic evaluators first.

## 8.1 Required metrics

### Tool Selection Accuracy (TSA)

Exact normalized comparison between predicted and expected tool name.

```text
1.0 = correct tool
0.0 = incorrect/missing tool
```

### Argument Accuracy (AA)

Score expected arguments against predicted arguments.

Requirements:

- exact key/value correctness by default,
- configurable tolerant scalar normalization for obvious representation differences,
- missing expected args reduce score,
- extra args should be surfaced and optionally penalized,
- never use an LLM judge for this metric.

### Composite score

Use an explicit configurable formula. Initial default:

```text
0.5 * TSA + 0.5 * AA
```

If the wrong tool is selected, make the scoring behavior explicit. A reasonable default is to return 0 for the complete tool-call score because argument correctness for the wrong tool is not meaningful.

### Regression pass rate

Measure the percentage of regression cases meeting expected behavior.

### Cost and latency

Track when data is available, but do not block v0.1 correctness if provider usage metadata is incomplete.

## 8.2 Evaluator output

Return structured records, not just floats:

```python
@dataclass(frozen=True)
class EvalResult:
    total_score: float
    tool_selection_score: float
    argument_accuracy_score: float
    passed: bool
    reason: str
```

The optimizer may consume `total_score`; reports should expose all submetrics.

---

# 9. Experimental arms

The repository is incomplete unless all three arms are implemented and comparable.

## 9.1 Arm A — Original agent

Evaluate the untouched initial artifacts.

Persist:

- per-case predictions,
- per-case metrics,
- aggregate metrics,
- cost/latency summaries when available,
- artifact hashes.

## 9.2 Arm B — Single-shot repair

Input to the repair model:

- a concise diagnosis of the known failure cluster,
- failing optimization examples or summarized evidence,
- current editable artifacts,
- allowed edit surfaces.

Ask for one best repair.

Rules:

- exactly one repair proposal,
- same edit budget as the optimizer where practical,
- evaluate on optimization split for reporting,
- evaluate on held-out and regression splits,
- do not hand-tune the output.

This arm is the key baseline.

## 9.3 Arm C — Optimizer-backed repair search

Implement candidate search over the same allowed edit surfaces.

Initial allowed surfaces:

1. system prompt,
2. tool descriptions.

The search must:

- generate multiple candidates,
- score candidates only on the optimization split,
- use evaluation feedback to propose improved candidates,
- keep a candidate history,
- have explicit budgets,
- select one finalist without consulting held-out labels,
- evaluate the finalist on held-out and regression splits.

Optimizer path:

1. Use the official public `gepa` package.
2. Inspect the actual package/API rather than guessing function signatures.
3. Pin the version used through the project lockfile.
4. Wrap it behind an internal adapter.
5. If GEPA is unavailable or incompatible, fail clearly with an actionable error.
   Do not introduce an alternate runtime optimizer path.

The repository must clearly record which optimizer actually ran.

---

# 10. Repair representation

Represent editable artifacts and candidate repairs explicitly.

Suggested types:

```python
@dataclass(frozen=True)
class AgentArtifacts:
    system_prompt: str
    tool_descriptions: dict[str, str]

@dataclass(frozen=True)
class RepairCandidate:
    candidate_id: str
    artifacts: AgentArtifacts
    parent_id: str | None
    generation: int
    rationale: str | None
```

Requirements:

- candidates must be serializable,
- each candidate must have a stable ID/hash,
- preserve lineage,
- preserve exact artifact text,
- produce a unified diff against baseline artifacts,
- avoid mutating baseline files during search.

---

# 11. Search budgets and fairness

The comparison must be honest.

Record:

- number of repair-model calls,
- number of agent evaluation calls,
- total examples evaluated,
- token usage when available,
- wall-clock duration.

Provide explicit CLI/config budgets, for example:

```text
max_candidates
max_generations
max_reflection_calls
max_eval_calls
```

Do not hide the fact that optimizer-backed search uses more compute than single-shot repair.

The README should present both quality and compute tradeoffs.

---

# 12. Validation and release gates

A candidate is not considered a successful repair merely because optimization-split score improves.

## 12.1 Held-out validation

After selecting the finalist:

- evaluate on held-out set,
- report absolute score,
- report delta from original,
- report delta from single-shot repair,
- include per-category breakdown.

## 12.2 Regression gate

Evaluate on the regression set.

Initial configurable gate:

```text
candidate_regression_score >= baseline_regression_score - tolerance
```

Default tolerance should be conservative and explicit.

A finalist that improves the target failure cluster but breaks legitimate refund behavior should fail the gate.

## 12.3 Patch output

For each repair arm produce:

- exact final artifact snapshot,
- unified diff against baseline,
- structured metrics,
- human-readable report.

---

# 13. CLI requirements

Provide a small CLI. Prefer a lightweight library such as Typer only if it meaningfully improves usability; otherwise use `argparse` and avoid unnecessary dependencies.

Target commands:

```bash
uv run agent-repair baseline
uv run agent-repair single-shot
uv run agent-repair optimize
uv run agent-repair compare
```

A convenience command is desirable:

```bash
uv run agent-repair run-all
```

`run-all` should:

1. validate configuration,
2. run original baseline,
3. run single-shot repair,
4. run optimizer-backed search,
5. evaluate finalists,
6. enforce regression gate,
7. write a comparison report.

Also support a cheap smoke mode:

```bash
uv run agent-repair run-all --optimizer gepa --smoke
```

Smoke mode should use a very small subset and tiny search budget.

---

# 14. Results and artifact layout

Write each run to a timestamped directory:

```text
runs/<run-id>/
├── config.json
├── environment.json
├── split_hashes.json
├── baseline/
│   ├── predictions.jsonl
│   ├── metrics.json
│   └── artifacts.json
├── single_shot/
│   ├── candidate.json
│   ├── diff.patch
│   ├── predictions.jsonl
│   └── metrics.json
├── optimizer/
│   ├── candidates.jsonl
│   ├── lineage.json
│   ├── finalist.json
│   ├── diff.patch
│   ├── predictions.jsonl
│   └── metrics.json
└── report.md
```

Do not commit large raw API payloads or secrets.

Commit one small representative result run only after it is real and reproducible.

---

# 15. Suggested repository structure

Prefer this shape unless implementation evidence justifies a small change:

```text
.
├── AGENTS.md
├── PLANS.md
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
├── src/
│   └── agent_repair/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── anthropic_client.py
│       ├── artifacts.py
│       ├── agent.py
│       ├── evaluator.py
│       ├── datasets.py
│       ├── reporting.py
│       ├── patches.py
│       └── repair/
│           ├── __init__.py
│           ├── base.py
│           ├── single_shot.py
│           └── gepa_adapter.py
├── agent/
│   ├── system_prompt.md
│   └── tools.json
├── evals/
│   ├── optimize_train.jsonl
│   ├── optimize_val.jsonl
│   ├── heldout.jsonl
│   └── regression.jsonl
├── experiments/
│   └── run_comparison.py
├── tests/
│   ├── test_evaluator.py
│   ├── test_datasets.py
│   ├── test_artifacts.py
│   └── test_smoke.py
└── runs/
    └── .gitkeep
```

Avoid over-fragmentation. Small modules with clear responsibilities are preferable to a deep framework architecture.

---

# 16. Engineering standards

## 16.1 Python

- Target Python 3.11+ unless a dependency requires a different supported version.
- Use type hints for public functions and core data structures.
- Prefer dataclasses or Pydantic only where validation warrants the dependency.
- Avoid `Any` in core interfaces when a real type is practical.
- Keep side effects at the edges.
- Keep experiment logic deterministic where possible.
- Seed any local random number generator and record the seed.

## 16.2 Dependency policy

- Minimize production dependencies.
- Before adding a dependency, verify that it removes meaningful code or risk.
- Pin direct dependencies in `pyproject.toml` using sensible compatible ranges and commit `uv.lock`.
- Never invent a third-party API from memory. Inspect the installed package or official docs.

## 16.3 Error handling

- Fail loudly on malformed eval data.
- Fail clearly on missing environment variables.
- Distinguish provider errors from evaluator errors.
- Do not swallow exceptions with broad `except Exception: pass` patterns.
- Include actionable context in errors without leaking secrets.

## 16.4 Logging

- Use standard logging, not scattered prints, for library internals.
- CLI may print concise progress and final summaries.
- Never log API keys or full sensitive headers.

---

# 17. Testing requirements

Before considering a change complete, run all applicable checks.

Expected commands should be implemented in the project and documented:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

If static typing is configured:

```bash
uv run mypy src
```

Required tests:

## 17.1 Evaluator unit tests

Cover at least:

- correct tool + correct args,
- correct tool + wrong args,
- wrong tool,
- missing tool,
- missing required arg,
- extra arg behavior,
- tolerant scalar normalization,
- multiple args with partial match.

## 17.2 Dataset tests

Cover:

- valid JSONL,
- duplicate IDs,
- malformed expected args,
- split overlap detection,
- deterministic split hashes.

## 17.3 Artifact tests

Cover:

- serialization,
- stable hashing,
- diff generation,
- baseline immutability.

## 17.4 Tests and smoke runs

Normal tests must avoid external API calls by using local stubs/spies at test
boundaries. Do not expose a product/runtime model-simulation path.

Live Anthropic integration tests must be opt-in, e.g.:

```bash
RUN_LIVE_ANTHROPIC_TESTS=1 uv run pytest -m live
```

Do not make normal unit tests spend API credits.

---

# 18. Experiment integrity

This repository is an empirical prototype. Treat result integrity as a core feature.

Rules:

- Never fabricate scores.
- Never write README results before a real run exists.
- Never manually edit generated metrics to improve the story.
- Never select candidates using held-out labels.
- Record configurations and artifact hashes.
- Preserve raw per-case prediction/eval records necessary to audit aggregate scores.
- Clearly label smoke-run numbers as smoke-run numbers.
- If optimizer-backed repair loses to single-shot repair, report that honestly.
- If a run is inconclusive, say so.

---

# 19. README requirements

The README should be concise, technical, and evidence-first.

Recommended structure:

1. Hero line.
2. One-sentence research question.
3. 20-second architecture diagram.
4. Quickstart.
5. Experimental arms.
6. Dataset split policy.
7. Metrics.
8. Results table populated only from real committed artifacts.
9. Example winning diff.
10. Reproduce command.
11. Limitations.
12. Roadmap.

Hard rules:

- No mention of LangChain.
- No mention of LangSmith.
- No unverified novelty claims.
- No internal employer references.
- No placeholder badges.
- No placeholder benchmark numbers presented as real.

---

# 20. Git and GitHub workflow

This repository must have a clean, meaningful history.

## 20.1 Initialize Git

If the directory is not already a Git repository:

```bash
git init
```

Use `main` as the primary branch.

## 20.2 Commit discipline

Create logical commits during implementation. Do not wait until the end for one giant commit.

Suggested milestones:

1. `chore: scaffold optimizer-backed agent repair experiment`
2. `feat: add tool-calling demo agent and eval datasets`
3. `feat: add deterministic tool-call evaluation`
4. `feat: add single-shot repair baseline`
5. `feat: add optimizer-backed repair search`
6. `feat: add held-out validation and regression gates`
7. `docs: add reproducible experiment guide and results`

Exact commit boundaries may differ, but history should tell the implementation story.

Before each commit:

- inspect `git diff`,
- ensure no secret is staged,
- run relevant tests,
- use an accurate commit message.

Never commit `.env`, API keys, raw secret-bearing payloads, or local virtual environments.

## 20.3 Remote repository

At the end of the build:

1. Check GitHub CLI availability:

```bash
gh --version
```

2. Check authentication:

```bash
gh auth status
```

3. If authenticated and authorized, create a public GitHub repository from the current directory and push `main`.

Preferred repository name:

```text
agent-repair-search
```

Suggested command pattern:

```bash
gh repo create agent-repair-search \
  --public \
  --source=. \
  --remote=origin \
  --push
```

If that name is unavailable, choose a close descriptive alternative and report the exact remote URL.

If authentication or permissions block remote creation, do not fabricate success. Complete all local Git work, preserve commits, and report the exact blocker and exact command the user must run.

After pushing:

```bash
git status
git log --oneline --decorate -n 10
git remote -v
```

The final state should be clean and synchronized with `origin/main` when permissions allow.

---

# 21. Planning protocol for coding agents

This build is non-trivial. Use `PLANS.md` as the execution ledger.

Before implementation:

1. inspect the working directory,
2. read `AGENTS.md`,
3. create or update `PLANS.md`,
4. record architecture decisions,
5. record milestones,
6. mark risks and unknown external APIs,
7. begin implementation only after the plan is coherent.

During implementation:

- update milestone status,
- record important deviations,
- record commands used for validation,
- do not turn `PLANS.md` into a verbose diary.

At completion:

- ensure `PLANS.md` reflects actual state,
- leave no false "done" items,
- summarize remaining limitations.

---

# 22. Execution behavior

When asked to implement the repository end-to-end:

1. Do not ask unnecessary clarification questions.
2. Make reasonable, explicit decisions that preserve the research question.
3. Inspect current public package APIs before integrating them.
4. Prefer a working narrow experiment over a broad unfinished framework.
5. Run tests frequently.
6. Surface blockers immediately, then continue on unblocked work.
7. Never claim a live experiment succeeded without actually running it.
8. If API credits or credentials are unavailable, complete the full local implementation with stubbed tests and leave exact live-run commands.
9. Do not stop after scaffolding.
10. Continue until the definition of done is met or an external permission/credential blocker is genuinely unavoidable.

---

# 23. Definition of done

The repository is complete for v0.1 only when all of the following are true:

- [ ] Git repository initialized on `main`.
- [ ] `uv` project configured and lockfile committed.
- [ ] Anthropic client abstraction implemented.
- [ ] No model identifier hardcoded; task and repair model roles configurable.
- [ ] Synthetic tool-using agent implemented.
- [ ] Five tool schemas implemented.
- [ ] Baseline system prompt and intentionally ambiguous tool descriptions committed.
- [ ] Optimization-train, optimization-validation, held-out, and regression datasets committed.
- [ ] Split overlap/leakage checks implemented.
- [ ] Tool selection evaluator implemented and tested.
- [ ] Argument evaluator implemented and tested.
- [ ] Composite metrics implemented.
- [ ] Original baseline arm implemented.
- [ ] Single-shot repair arm implemented.
- [ ] Optimizer-backed search arm implemented.
- [ ] Search budgets and candidate lineage recorded.
- [ ] Held-out evaluation implemented.
- [ ] Regression gate implemented.
- [ ] Unified patch output implemented.
- [ ] `run-all` or equivalent one-command workflow implemented.
- [ ] Offline smoke test passes without API calls.
- [ ] Formatting/lint/tests pass.
- [ ] README contains no mention of LangChain or LangSmith.
- [ ] README contains no fabricated results.
- [ ] At least one real result run committed if credentials and budget allow.
- [ ] Logical Git commits created.
- [ ] Remote public GitHub repo created and pushed when `gh` auth permits.
- [ ] Final worktree clean.

---

# 24. Final reporting contract

At the end of an implementation session, report:

1. what was built,
2. exact test/lint results,
3. whether a live Anthropic run was executed,
4. actual experiment numbers if available,
5. any failed or inconclusive results,
6. final Git commit SHAs,
7. remote repository URL if created,
8. exact remaining blockers.

Be factual. Do not embellish.
