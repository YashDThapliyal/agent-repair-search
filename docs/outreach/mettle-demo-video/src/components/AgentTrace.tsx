import React from "react";
import { interpolate } from "remotion";
import { COLORS, FONT, MONO } from "../theme";
import { reveal } from "../util";

export type TraceStatus = "pending" | "ok" | "fail" | "stale";

export type TraceNode = {
  label: string;
  status: TraceStatus;
};

const statusColor = (s: TraceStatus) =>
  s === "ok"
    ? COLORS.cyan
    : s === "fail"
      ? COLORS.red
      : s === "stale"
        ? COLORS.amber
        : COLORS.textFaint;

// A left-to-right agent trajectory of small graph nodes connected by a line.
// Nodes reveal in sequence; each node's dot and label fade in from the left.
// Used in the opening scene to show tool calls succeeding then a routing miss.
export const AgentTrace: React.FC<{
  nodes: TraceNode[];
  frame: number;
  startFrame?: number;
  stepFrames?: number;
  cx: number; // center x of the row
  cy: number; // baseline y of the dots
  gap?: number;
}> = ({
  nodes,
  frame,
  startFrame = 0,
  stepFrames = 16,
  cx,
  cy,
  gap = 150,
}) => {
  const total = (nodes.length - 1) * gap;
  const x0 = cx - total / 2;

  return (
    <>
      {/* connecting segments (drawn behind dots) */}
      <svg
        width={1920}
        height={1080}
        viewBox="0 0 1920 1080"
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
      >
        {nodes.slice(1).map((n, i) => {
          const segStart = startFrame + (i + 1) * stepFrames - 6;
          const p = reveal(frame, segStart, 12);
          const xa = x0 + i * gap;
          const xb = x0 + (i + 1) * gap;
          const drawnX = interpolate(p, [0, 1], [xa, xb]);
          const segColor =
            n.status === "fail" || n.status === "stale"
              ? statusColor(n.status)
              : "rgba(255,255,255,0.18)";
          return (
            <line
              key={i}
              x1={xa}
              y1={cy}
              x2={drawnX}
              y2={cy}
              stroke={segColor}
              strokeWidth={2.5}
              strokeLinecap="round"
              opacity={p}
            />
          );
        })}
      </svg>

      {nodes.map((n, i) => {
        const nodeStart = startFrame + i * stepFrames;
        const p = reveal(frame, nodeStart, 14);
        const x = x0 + i * gap;
        const color = statusColor(n.status);
        const active = n.status !== "pending";
        return (
          <React.Fragment key={i}>
            <div
              style={{
                position: "absolute",
                left: x - 17,
                top: cy - 17,
                width: 34,
                height: 34,
                borderRadius: 999,
                background: active ? `${color}22` : "transparent",
                border: `2.5px solid ${color}`,
                boxShadow:
                  n.status === "fail"
                    ? `0 0 22px ${COLORS.red}88`
                    : n.status === "stale"
                      ? `0 0 18px ${COLORS.amber}66`
                      : "none",
                opacity: p,
                scale: interpolate(p, [0, 1], [0.5, 1]),
              }}
            />
            <div
              style={{
                position: "absolute",
                left: x - 90,
                top: cy + 32,
                width: 180,
                textAlign: "center",
                fontFamily: MONO,
                fontSize: 20,
                fontWeight: 500,
                color:
                  n.status === "pending"
                    ? COLORS.textFaint
                    : n.status === "ok"
                      ? COLORS.textDim
                      : color,
                opacity: interpolate(p, [0.3, 1], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                }),
              }}
            >
              {n.label}
            </div>
          </React.Fragment>
        );
      })}
    </>
  );
};

// Small helper: a status glyph (check / cross) for inline use.
export const StatusGlyph: React.FC<{ ok: boolean; size?: number }> = ({
  ok,
  size = 26,
}) => (
  <span
    style={{
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      width: size,
      height: size,
      borderRadius: 999,
      fontFamily: FONT,
      fontSize: size * 0.62,
      fontWeight: 800,
      color: ok ? COLORS.green : COLORS.red,
      background: ok ? COLORS.greenSoft : COLORS.redSoft,
      border: `1.5px solid ${ok ? COLORS.green : COLORS.red}`,
    }}
  >
    {ok ? "✓" : "✕"}
  </span>
);
