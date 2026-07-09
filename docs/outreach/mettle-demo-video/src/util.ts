import { Easing, interpolate } from "remotion";
import { EASE_OUT } from "./theme";

// A clamped interpolation with a Bézier ease-out entrance — the workhorse
// reveal used across scenes. Returns a 0..1 progress value.
export const reveal = (
  frame: number,
  start: number,
  duration = 18,
  ease: readonly [number, number, number, number] = EASE_OUT,
): number =>
  interpolate(frame, [start, start + duration], [0, 1], {
    easing: Easing.bezier(ease[0], ease[1], ease[2], ease[3]),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

// Fade + slight upward drift, driven by a progress value (0..1).
export const riseStyle = (progress: number, distance = 24) => ({
  opacity: progress,
  translate: `0px ${interpolate(progress, [0, 1], [distance, 0])}px`,
});

// Symmetric in/out envelope: rises over `inDur`, holds, falls over `outDur`
// before `total`. Useful for captions that must appear and clear.
export const envelope = (
  frame: number,
  total: number,
  inDur = 14,
  outDur = 12,
): number => {
  const rise = interpolate(frame, [0, inDur], [0, 1], {
    easing: Easing.bezier(EASE_OUT[0], EASE_OUT[1], EASE_OUT[2], EASE_OUT[3]),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fall = interpolate(frame, [total - outDur, total], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return Math.min(rise, fall);
};
