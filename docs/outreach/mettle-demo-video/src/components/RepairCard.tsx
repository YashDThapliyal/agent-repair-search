import React from "react";
import { interpolate } from "remotion";
import { COLORS, FONT, MONO } from "../theme";
import { reveal, riseStyle } from "../util";

export type DiffLine = { kind: "add" | "del" | "ctx"; text: string };

// A repair-approach card: title, one-line description, an optional small
// diff-style patch preview, and an optional corner tag (e.g. "GEPA").
// Used in the RepairComparison scene for Original / Focused / Bounded search.
export const RepairCard: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  subtitle?: string;
  accent: "muted" | "indigo" | "cyan";
  tag?: string;
  diff?: DiffLine[];
  frame: number;
  startFrame: number;
  emphasis?: number;
}> = ({
  x,
  y,
  w,
  h,
  title,
  subtitle,
  accent,
  tag,
  diff,
  frame,
  startFrame,
  emphasis = 1,
}) => {
  const p = reveal(frame, startFrame, 22);
  const accentColor =
    accent === "indigo"
      ? COLORS.indigo
      : accent === "cyan"
        ? COLORS.cyan
        : COLORS.borderStrong;
  const isMuted = accent === "muted";

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: w,
        height: h,
        borderRadius: 18,
        boxSizing: "border-box",
        padding: "22px 24px",
        background: isMuted
          ? "rgba(255,255,255,0.02)"
          : "linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))",
        border: `1.5px solid ${
          isMuted ? COLORS.border : `${accentColor}${emphasis > 0.6 ? "AA" : "66"}`
        }`,
        boxShadow: isMuted
          ? "none"
          : `0 0 34px ${accentColor}${Math.round(emphasis * 34)
              .toString(16)
              .padStart(2, "0")}, 0 14px 34px rgba(0,0,0,0.3)`,
        display: "flex",
        flexDirection: "column",
        fontFamily: FONT,
        ...riseStyle(p, 24),
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div>
          <div
            style={{
              fontSize: 30,
              fontWeight: 700,
              color: isMuted ? COLORS.textDim : COLORS.text,
            }}
          >
            {title}
          </div>
          {subtitle && (
            <div
              style={{
                fontSize: 21,
                fontWeight: 500,
                color: COLORS.textDim,
                marginTop: 6,
              }}
            >
              {subtitle}
            </div>
          )}
        </div>
        {tag && (
          <span
            style={{
              fontFamily: MONO,
              fontSize: 18,
              fontWeight: 700,
              letterSpacing: 1,
              color: accentColor,
              padding: "5px 12px",
              borderRadius: 8,
              background: `${accentColor}1F`,
              border: `1px solid ${accentColor}66`,
            }}
          >
            {tag}
          </span>
        )}
      </div>

      {diff && (
        <div
          style={{
            marginTop: 18,
            padding: "14px 16px",
            borderRadius: 10,
            background: "rgba(0,0,0,0.30)",
            border: `1px solid ${COLORS.border}`,
            fontFamily: MONO,
            fontSize: 18,
            lineHeight: 1.55,
            flex: 1,
            overflow: "hidden",
          }}
        >
          {diff.map((line, i) => {
            const lineP = reveal(frame, startFrame + 12 + i * 5, 12);
            const c =
              line.kind === "add"
                ? COLORS.green
                : line.kind === "del"
                  ? COLORS.red
                  : COLORS.textFaint;
            const prefix =
              line.kind === "add" ? "+ " : line.kind === "del" ? "- " : "  ";
            return (
              <div
                key={i}
                style={{
                  color: c,
                  opacity: lineP,
                  translate: `${interpolate(lineP, [0, 1], [-8, 0])}px 0px`,
                  whiteSpace: "nowrap",
                }}
              >
                {prefix}
                {line.text}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
