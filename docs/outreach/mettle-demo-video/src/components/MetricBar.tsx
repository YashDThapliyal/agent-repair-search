import React from "react";
import { interpolate } from "remotion";
import { COLORS, FONT, MONO } from "../theme";
import { reveal } from "../util";

// A single labeled horizontal metric bar. The fill grows from 0 to `value`
// (0..1 scale) and the numeric readout counts up in lockstep, so the bar and
// the printed number always agree. Values come straight from the repository.
export const MetricBar: React.FC<{
  label: string;
  value: number; // 0..1
  frame: number;
  startFrame: number;
  color?: string;
  width: number;
  // Domain the bar fill maps across (defaults to full 0..1). Narrowing the
  // domain makes small differences legible without misstating the number.
  domain?: [number, number];
  highlight?: boolean;
  labelColor?: string;
  barHeight?: number;
}> = ({
  label,
  value,
  frame,
  startFrame,
  color = COLORS.indigo,
  width,
  domain = [0, 1],
  highlight = false,
  labelColor = COLORS.textDim,
  barHeight = 26,
}) => {
  const p = reveal(frame, startFrame, 28);
  const fillFrac = interpolate(value, domain, [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const shownFrac = fillFrac * p;
  const shownValue = value * p;

  return (
    <div style={{ width, fontFamily: FONT }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 10,
        }}
      >
        <span
          style={{
            fontSize: 26,
            fontWeight: highlight ? 700 : 600,
            color: highlight ? COLORS.text : labelColor,
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontFamily: MONO,
            fontSize: 30,
            fontWeight: 700,
            color: highlight ? color : COLORS.text,
          }}
        >
          {shownValue.toFixed(3)}
        </span>
      </div>
      <div
        style={{
          position: "relative",
          height: barHeight,
          borderRadius: 999,
          background: "rgba(255,255,255,0.06)",
          border: `1px solid ${COLORS.border}`,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            width: `${shownFrac * 100}%`,
            borderRadius: 999,
            background: highlight
              ? `linear-gradient(90deg, ${color}, ${color}CC)`
              : `${color}AA`,
            boxShadow: highlight ? `0 0 20px ${color}66` : "none",
          }}
        />
      </div>
    </div>
  );
};
