import React from "react";
import { interpolate } from "remotion";
import { COLORS, FONT } from "../theme";
import { reveal } from "../util";

const SIGNALS = [
  "failure type",
  "expected gain",
  "regression risk",
  "compute budget",
];

// The focal element of the Future Direction scene: a labeled selector box that
// ingests four decision signals. Deliberately the strongest visual weight in
// its scene. Marked as future direction by the surrounding scene, not here.
export const StrategySelector: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  frame: number;
  startFrame: number;
}> = ({ x, y, w, h, frame, startFrame }) => {
  const p = reveal(frame, startFrame, 26);
  const pulse =
    0.5 +
    0.5 *
      Math.sin(interpolate(frame, [0, 90], [0, Math.PI * 2]) % (Math.PI * 2));
  const glow = 0.28 + pulse * 0.14;

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: w,
        height: h,
        borderRadius: 26,
        boxSizing: "border-box",
        padding: "30px 30px 28px",
        background:
          "linear-gradient(180deg, rgba(139,124,240,0.16), rgba(139,124,240,0.06))",
        border: `2.5px solid ${COLORS.indigo}`,
        boxShadow: `0 0 60px rgba(139,124,240,${glow}), 0 20px 60px rgba(0,0,0,0.35)`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: FONT,
        opacity: p,
        scale: interpolate(p, [0, 1], [0.92, 1]),
      }}
    >
      <div
        style={{
          fontSize: 38,
          fontWeight: 800,
          color: COLORS.text,
          textAlign: "center",
          lineHeight: 1.05,
          letterSpacing: 0.2,
        }}
      >
        Repair Strategy
        <br />
        Selector
      </div>
      <div
        style={{
          fontSize: 20,
          fontWeight: 500,
          color: COLORS.textDim,
          marginTop: 10,
          textAlign: "center",
          maxWidth: 250,
          lineHeight: 1.3,
        }}
      >
        choose a repair approach from the diagnosis
      </div>

      <div
        style={{
          marginTop: 22,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 10,
          width: "100%",
        }}
      >
        {SIGNALS.map((s, i) => {
          const sp = reveal(frame, startFrame + 16 + i * 7, 14);
          return (
            <div
              key={s}
              style={{
                minHeight: 52,
                padding: "8px 6px",
                borderRadius: 12,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                background: "rgba(255,255,255,0.06)",
                border: `1px solid ${COLORS.indigo}55`,
                color: COLORS.indigo,
                fontSize: 18,
                fontWeight: 700,
                lineHeight: 1.15,
                opacity: sp,
                translate: `0px ${interpolate(sp, [0, 1], [10, 0])}px`,
              }}
            >
              {s}
            </div>
          );
        })}
      </div>
    </div>
  );
};
