// Centralized design tokens and timing constants for the Mettle demo video.
// All scientific values referenced in scenes are grounded in the repository:
//   reports/final_experiment.json  (composite mean scores, held-out target slice)
//   scenarios/stateful_account_resolution/scenario.json
// Do not alter the numeric values below without re-checking those sources.

export const FPS = 30;

// ---------------------------------------------------------------------------
// Color system: dark navy base, off-white text, muted indigo primary accent,
// restrained cyan secondary. Amber/red mark failure, muted green marks safety.
// ---------------------------------------------------------------------------
export const COLORS = {
  bg: "#0A0E1A",
  bgDeep: "#070A14",
  panel: "rgba(255, 255, 255, 0.035)",
  panelSolid: "#111726",
  border: "rgba(255, 255, 255, 0.10)",
  borderStrong: "rgba(255, 255, 255, 0.18)",

  text: "#EAECF5",
  textDim: "#9AA3B8",
  textFaint: "#5D667E",

  indigo: "#8B7CF0", // primary accent
  indigoDeep: "#6F5ED7",
  indigoSoft: "rgba(139, 124, 240, 0.14)",

  cyan: "#54C7DC", // secondary accent (restrained)
  cyanSoft: "rgba(84, 199, 220, 0.14)",

  amber: "#E4A85C", // caution / stale routing
  red: "#E56A73", // failure
  redSoft: "rgba(229, 106, 115, 0.16)",
  green: "#6FBAA6", // pass / safe / regression
  greenSoft: "rgba(111, 186, 166, 0.14)",
} as const;

export const FONT =
  'Inter, "SF Pro Display", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';
export const MONO =
  '"SF Mono", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace';

// ---------------------------------------------------------------------------
// Scene timeline (30 fps). Durations chosen to land the full video at ~65s,
// inside the requested 55-65s window.
// ---------------------------------------------------------------------------
export const SCENES = {
  opening: 180, // 0:00 - 0:06
  failure: 240, // 0:06 - 0:14
  repair: 315, // 0:14 - 0:24.5
  evaluation: 300, // 0:24.5 - 0:34.5
  results: 300, // 0:34.5 - 0:44.5
  future: 345, // 0:44.5 - 0:56
  connection: 150, // 0:56 - 1:01
  closing: 120, // 1:01 - 1:05
} as const;

export const TOTAL_FRAMES = Object.values(SCENES).reduce((a, b) => a + b, 0);

// Standard easing curves (from the Remotion timing best-practices).
// Crisp UI entrance — strong ease-out, no overshoot.
export const EASE_OUT = [0.16, 1, 0.3, 1] as const;
// Balanced editorial ease-in-out for slow reveals.
export const EASE_INOUT = [0.45, 0, 0.55, 1] as const;
// Gentle overshoot for emphasis (use sparingly).
export const EASE_POP = [0.34, 1.4, 0.64, 1] as const;

// ---------------------------------------------------------------------------
// Grounded experiment values (source: reports/final_experiment.json).
// composite_mean_score, rounded to the same precision used in reports/README.
// ---------------------------------------------------------------------------
export const RESULTS = {
  original: { optimizeVal: 0.6, heldout: 0.475, regression: 0.933 },
  focused: { optimizeVal: 0.733, heldout: 0.55, regression: 0.933 },
  gepa: { optimizeVal: 0.7, heldout: 0.5, regression: 0.956 },
} as const;
