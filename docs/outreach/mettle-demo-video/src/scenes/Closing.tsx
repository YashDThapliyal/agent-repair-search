import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { COLORS, FONT, MONO } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 8 (1:02-1:05): closing.
// Repo URL is the confirmed live public repository:
// github.com/YashDThapliyal/mettle-agent-repair-search

export const Closing: React.FC = () => {
  const frame = useCurrentFrame();

  const nameP = reveal(frame, 4, 20);
  const subP = reveal(frame, 20, 18);
  const lineP = reveal(frame, 36, 20);
  const urlP = reveal(frame, 54, 18);

  return (
    <AbsoluteFill>
      <Background glow="indigo" />

      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: FONT,
        }}
      >
        <div
          style={{
            fontSize: 150,
            fontWeight: 850,
            letterSpacing: -2,
            color: COLORS.text,
            ...riseStyle(nameP, 24),
          }}
        >
          Mettle
        </div>

        <div
          style={{
            fontSize: 40,
            fontWeight: 600,
            letterSpacing: 6,
            textTransform: "uppercase",
            color: COLORS.indigo,
            marginTop: 6,
            opacity: subP,
          }}
        >
          Agent Repair Search
        </div>

        <div
          style={{
            width: 120,
            height: 2,
            background: COLORS.border,
            margin: "42px 0",
            opacity: lineP,
          }}
        />

        <div
          style={{
            fontSize: 40,
            fontWeight: 550,
            color: COLORS.textDim,
            maxWidth: 1200,
            textAlign: "center",
            ...riseStyle(lineP, 18),
          }}
        >
          Choosing how to repair, not just how hard to search.
        </div>

        <div
          style={{
            fontFamily: MONO,
            fontSize: 27,
            color: COLORS.text,
            marginTop: 64,
            padding: "12px 26px",
            borderRadius: 12,
            background: "rgba(255,255,255,0.04)",
            border: `1px solid ${COLORS.border}`,
            opacity: urlP,
          }}
        >
          github.com/YashDThapliyal/mettle-agent-repair-search
        </div>

        <div
          style={{
            fontSize: 24,
            fontWeight: 600,
            color: COLORS.textFaint,
            marginTop: 26,
            opacity: urlP,
          }}
        >
          Yash D. Thapliyal
        </div>
      </div>
    </AbsoluteFill>
  );
};
