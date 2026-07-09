import React from "react";
import { interpolate } from "remotion";
import { COLORS, FONT } from "../theme";
import { reveal, riseStyle } from "../util";

// Small CSS/SVG lock, drawn (no external icon assets).
export const LockIcon: React.FC<{ size?: number; color?: string }> = ({
  size = 34,
  color = COLORS.cyan,
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <rect
      x="4"
      y="10.5"
      width="16"
      height="10.5"
      rx="2.5"
      stroke={color}
      strokeWidth="1.8"
      fill={`${color}22`}
    />
    <path
      d="M7.5 10.5 V7.5 a4.5 4.5 0 0 1 9 0 V10.5"
      stroke={color}
      strokeWidth="1.8"
      fill="none"
    />
    <circle cx="12" cy="15.5" r="1.6" fill={color} />
  </svg>
);

export type GateStage = {
  index: string;
  title: string;
  sub: string;
  accent: "indigo" | "cyan" | "green";
  locked?: boolean;
};

// A single evaluation stage card. Reused in the Evaluation scene as a
// three-stage protocol: Optimize -> Untouched held-out -> Regression.
export const EvalStageCard: React.FC<{
  stage: GateStage;
  frame: number;
  startFrame: number;
  w: number;
  h: number;
}> = ({ stage, frame, startFrame, w, h }) => {
  const p = reveal(frame, startFrame, 20);
  const accentColor =
    stage.accent === "cyan"
      ? COLORS.cyan
      : stage.accent === "green"
        ? COLORS.green
        : COLORS.indigo;

  return (
    <div
      style={{
        width: w,
        height: h,
        borderRadius: 20,
        boxSizing: "border-box",
        padding: "30px 30px",
        background: `linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))`,
        border: `1.5px solid ${accentColor}66`,
        boxShadow: `0 14px 40px rgba(0,0,0,0.28)`,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        fontFamily: FONT,
        ...riseStyle(p, 26),
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div
          style={{
            width: 44,
            height: 44,
            borderRadius: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: `${accentColor}22`,
            border: `1.5px solid ${accentColor}`,
            color: accentColor,
            fontWeight: 800,
            fontSize: 24,
          }}
        >
          {stage.index}
        </div>
        {stage.locked && <LockIcon size={34} color={accentColor} />}
      </div>
      <div>
        <div
          style={{
            fontSize: 32,
            fontWeight: 700,
            color: COLORS.text,
            marginBottom: 8,
            lineHeight: 1.1,
          }}
        >
          {stage.title}
        </div>
        <div style={{ fontSize: 23, fontWeight: 500, color: COLORS.textDim }}>
          {stage.sub}
        </div>
      </div>
    </div>
  );
};

// A small badge, e.g. "Held-out never reused" (grounded: heldout_reused=false).
export const Badge: React.FC<{
  children: React.ReactNode;
  frame: number;
  startFrame: number;
  color?: string;
}> = ({ children, frame, startFrame, color = COLORS.cyan }) => {
  const p = reveal(frame, startFrame, 18);
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        padding: "12px 22px",
        borderRadius: 999,
        background: `${color}18`,
        border: `1.5px solid ${color}`,
        color,
        fontFamily: FONT,
        fontSize: 24,
        fontWeight: 600,
        opacity: p,
        scale: interpolate(p, [0, 1], [0.9, 1]),
      }}
    >
      <LockIcon size={24} color={color} />
      {children}
    </div>
  );
};
