import React from "react";
import { COLORS, FONT, MONO } from "../theme";

// Absolutely-positioned rounded technical card in native 1920x1080 space.
// Scenes place these at explicit x/y/w/h so arrows (FlowArrow) can connect to
// known anchor points. `accent` tints the border and adds a restrained glow.
export const Card: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  accent?: "indigo" | "cyan" | "red" | "amber" | "green" | "muted";
  emphasis?: number; // 0..1, scales glow / border strength
  style?: React.CSSProperties;
  children?: React.ReactNode;
}> = ({ x, y, w, h, accent = "muted", emphasis = 1, style, children }) => {
  const accentColor = {
    indigo: COLORS.indigo,
    cyan: COLORS.cyan,
    red: COLORS.red,
    amber: COLORS.amber,
    green: COLORS.green,
    muted: COLORS.borderStrong,
  }[accent];

  const isMuted = accent === "muted";

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: w,
        height: h,
        borderRadius: 20,
        background: isMuted
          ? "rgba(255,255,255,0.025)"
          : `linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))`,
        border: `1.5px solid ${
          isMuted
            ? COLORS.border
            : `rgba(${hexToRgb(accentColor)}, ${0.35 + emphasis * 0.35})`
        }`,
        boxShadow: isMuted
          ? "0 12px 30px rgba(0,0,0,0.25)"
          : `0 0 0 1px rgba(${hexToRgb(accentColor)}, ${emphasis * 0.12}), 0 12px 40px rgba(${hexToRgb(accentColor)}, ${emphasis * 0.14})`,
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        ...style,
      }}
    >
      {children}
    </div>
  );
};

export const Pill: React.FC<{
  children: React.ReactNode;
  color?: string;
  bg?: string;
  mono?: boolean;
  size?: number;
  style?: React.CSSProperties;
}> = ({ children, color = COLORS.textDim, bg, mono, size = 22, style }) => (
  <span
    style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 8,
      padding: `7px 16px`,
      borderRadius: 999,
      fontFamily: mono ? MONO : FONT,
      fontSize: size,
      fontWeight: 600,
      letterSpacing: mono ? 0 : 0.2,
      color,
      background: bg ?? "rgba(255,255,255,0.05)",
      border: `1px solid ${COLORS.border}`,
      whiteSpace: "nowrap",
      ...style,
    }}
  >
    {children}
  </span>
);

// Small label used above sections, e.g. "Completed experiment".
export const Kicker: React.FC<{
  children: React.ReactNode;
  color?: string;
  style?: React.CSSProperties;
}> = ({ children, color = COLORS.indigo, style }) => (
  <div
    style={{
      fontFamily: FONT,
      fontSize: 22,
      fontWeight: 700,
      letterSpacing: 3,
      textTransform: "uppercase",
      color,
      ...style,
    }}
  >
    {children}
  </div>
);

function hexToRgb(hex: string): string {
  const m = hex.replace("#", "");
  const r = parseInt(m.substring(0, 2), 16);
  const g = parseInt(m.substring(2, 4), 16);
  const b = parseInt(m.substring(4, 6), 16);
  return `${r}, ${g}, ${b}`;
}
