# Mettle demo video (Remotion)

A ~65s programmatic technical explainer for the Mettle experiment
(github.com/YashDThapliyal/mettle-agent-repair-search), built with Remotion
(React + TypeScript + SVG, frame-based animation). Renders to
`out/mettle-demo.mp4`. No audio.

## What it covers

1. The problem — an agent fails; what should repair it?
2. The studied failure — state-dependent tool routing (`stateful_account_resolution`).
3. What Mettle compares — Original vs Focused (single-shot) repair vs Bounded GEPA search.
4. The evaluation protocol — Optimize / untouched held-out / Regression.
5. The result — the exact composite mean scores.
6. Future direction — a Repair Strategy Selector (explicitly *not implemented*).
7. Connection to broader agent-improvement systems (kept general).
8. Closing.

## Fact sources (do not edit numbers without re-checking these)

- `../../reports/final_experiment.json` — composite mean scores, held-out target slice.
- `../../reports/final_experiment.md`, `benchmark_readiness.md`, `scenario_selection.md`.
- `../../scenarios/stateful_account_resolution/scenario.json` — diagnosis, splits, families.
- `../../README.md` — arm definitions, search accounting, future direction.

All numbers live in `src/theme.ts` (`RESULTS`). The video reruns no experiment.

## Render

```bash
npm install
npx remotion studio                       # preview
npx remotion render MettleDemo out/mettle-demo.mp4 --codec=h264 --muted --crf=18
```

- 1920×1080, 30 fps, H.264, ~65s, no audio.
- Official Remotion agent skill (`remotion-best-practices`) informed the build.
- No external/remote assets or fonts; local/system font stack only.

## Structure

- `src/theme.ts` — colors, timing, and grounded result values.
- `src/MettleDemo.tsx` — composition (8 scenes via `Series`, short scene fades).
- `src/scenes/` — one file per scene.
- `src/components/` — reusable pieces (AgentTrace, FlowArrow, RepairCard,
  EvalGate, MetricBar, StrategySelector, Background, primitives).
