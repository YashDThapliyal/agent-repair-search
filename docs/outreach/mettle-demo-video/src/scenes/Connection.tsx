import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { FlowArrow } from "../components/FlowArrow";
import { COLORS, FONT } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 7 (0:57-1:02): connection to broader agent-improvement systems.
// Deliberately general and non-presumptuous. Names no product and implies no
// partnership. The new middle layer (repair-strategy choice) is highlighted;
// the surrounding diagnose/eval stages are shown as existing context.

const STAGES = [
  { title: "Diagnose failures", accent: "muted" as const, w: 300 },
  { title: "Build targeted evals", accent: "muted" as const, w: 300 },
  { title: "Which repair strategy?", accent: "indigo" as const, w: 360 },
  { title: "Evaluate & deploy", accent: "muted" as const, w: 300 },
];

export const Connection: React.FC = () => {
  const frame = useCurrentFrame();

  const topP = reveal(frame, 4, 20);
  const qP = reveal(frame, 66, 22);
  const subP = reveal(frame, 92, 20);

  // pipeline geometry
  const gap = 44;
  const totalW =
    STAGES.reduce((a, s) => a + s.w, 0) + gap * (STAGES.length - 1);
  let cursor = (1920 - totalW) / 2;
  const boxes = STAGES.map((s) => {
    const x = cursor;
    cursor += s.w + gap;
    return { ...s, x };
  });
  const cy = 430;
  const h = 116;

  return (
    <AbsoluteFill>
      <Background glow="indigo" />

      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 236,
          textAlign: "center",
          fontFamily: FONT,
          fontSize: 38,
          fontWeight: 550,
          color: COLORS.textDim,
          ...riseStyle(topP, 16),
        }}
      >
        For systems that already diagnose failures and build targeted evals…
      </div>

      {/* arrows */}
      {boxes.slice(1).map((b, i) => (
        <FlowArrow
          key={i}
          id={`cn${i}`}
          from={{ x: boxes[i].x + boxes[i].w, y: cy }}
          to={{ x: b.x, y: cy }}
          progress={reveal(frame, 30 + i * 8, 16)}
          color={i === 1 ? COLORS.indigo : COLORS.textFaint}
          width={2.6}
        />
      ))}

      {/* stage boxes */}
      {boxes.map((b, i) => {
        const p = reveal(frame, 18 + i * 8, 18);
        const isNew = b.accent === "indigo";
        const pulse = isNew
          ? 0.5 + 0.5 * Math.sin(interpolate(frame, [0, 60], [0, Math.PI * 2]))
          : 0;
        return (
          <div
            key={b.title}
            style={{
              position: "absolute",
              left: b.x,
              top: cy - h / 2,
              width: b.w,
              height: h,
              borderRadius: 16,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              textAlign: "center",
              padding: "0 22px",
              boxSizing: "border-box",
              background: isNew
                ? "linear-gradient(180deg, rgba(139,124,240,0.18), rgba(139,124,240,0.06))"
                : "rgba(255,255,255,0.025)",
              border: `${isNew ? 2.5 : 1.5}px solid ${
                isNew ? COLORS.indigo : COLORS.border
              }`,
              boxShadow: isNew
                ? `0 0 ${28 + pulse * 20}px rgba(139,124,240,0.4)`
                : "none",
              color: isNew ? COLORS.text : COLORS.textDim,
              fontFamily: FONT,
              fontSize: isNew ? 30 : 26,
              fontWeight: isNew ? 750 : 600,
              ...riseStyle(p, 16),
            }}
          >
            {b.title}
          </div>
        );
      })}

      {/* the question */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 610,
          textAlign: "center",
          fontFamily: FONT,
        }}
      >
        <div
          style={{
            fontSize: 62,
            fontWeight: 800,
            color: COLORS.text,
            letterSpacing: -0.5,
            ...riseStyle(qP, 22),
          }}
        >
          Which repair strategy should run next?
        </div>
        <div
          style={{
            fontSize: 32,
            fontWeight: 550,
            color: COLORS.indigo,
            marginTop: 22,
            opacity: subP,
          }}
        >
          Choose based on the failure, risk, and budget.
        </div>
      </div>
    </AbsoluteFill>
  );
};
