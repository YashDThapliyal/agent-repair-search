import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { FlowArrow } from "../components/FlowArrow";
import { StrategySelector } from "../components/StrategySelector";
import { COLORS, FONT, MONO } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 6 (0:45-0:57): the proposed next layer.
// Explicitly future direction — grounded in README "Future Direction:
// Budget-Aware Repair Strategy Selection ... future work, not implemented in
// Mettle." The Repair Strategy Selector is the visual focal point; upstream
// diagnosis/eval stages are shown dimmed as existing context.

type NodeSpec = {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  tag?: string;
  accent: "muted" | "indigo" | "cyan";
  checks?: string[];
};

const NodeCard: React.FC<{
  n: NodeSpec;
  frame: number;
  startFrame: number;
  dim?: boolean;
}> = ({ n, frame, startFrame, dim }) => {
  const p = reveal(frame, startFrame, 20);
  const accent =
    n.accent === "indigo"
      ? COLORS.indigo
      : n.accent === "cyan"
        ? COLORS.cyan
        : COLORS.borderStrong;
  const muted = n.accent === "muted";
  return (
    <div
      style={{
        position: "absolute",
        left: n.x,
        top: n.y,
        width: n.w,
        height: n.h,
        borderRadius: 16,
        boxSizing: "border-box",
        padding: "18px 22px",
        background: muted
          ? "rgba(255,255,255,0.025)"
          : `linear-gradient(180deg, ${accent}18, ${accent}07)`,
        border: `1.5px solid ${muted ? COLORS.border : `${accent}99`}`,
        boxShadow: muted ? "none" : `0 0 26px ${accent}1F`,
        fontFamily: FONT,
        opacity: (dim ? 0.82 : 1) * p,
        translate: riseStyle(p, 16).translate,
        display: "flex",
        flexDirection: "column",
        justifyContent: n.checks ? "flex-start" : "center",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div
          style={{
            fontSize: 27,
            fontWeight: 720,
            color: muted ? COLORS.textDim : COLORS.text,
            lineHeight: 1.1,
          }}
        >
          {n.title}
        </div>
        {n.tag && (
          <span
            style={{
              fontFamily: MONO,
              fontSize: 15,
              fontWeight: 700,
              color: accent,
              padding: "3px 9px",
              borderRadius: 7,
              background: `${accent}1F`,
              border: `1px solid ${accent}66`,
            }}
          >
            {n.tag}
          </span>
        )}
      </div>
      {n.sub && (
        <div style={{ fontSize: 19, color: COLORS.textDim, marginTop: 7 }}>
          {n.sub}
        </div>
      )}
      {n.checks && (
        <div
          style={{
            marginTop: 12,
            display: "flex",
            flexDirection: "column",
            gap: 7,
          }}
        >
          {n.checks.map((c, i) => {
            const cp = reveal(frame, startFrame + 10 + i * 5, 10);
            return (
              <div
                key={c}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 9,
                  fontSize: 18,
                  color: COLORS.textDim,
                  opacity: cp,
                }}
              >
                <span
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: 999,
                    background: COLORS.green,
                  }}
                />
                {c}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

const STRAT = [
  { title: "Focused repair", sub: "one targeted change", y: 324, h: 120 },
  {
    title: "Bounded search",
    sub: "search over candidates",
    tag: "GEPA",
    y: 476,
    h: 140,
  },
  { title: "Other strategy", sub: "future methods", y: 648, h: 120 },
];

export const FutureDirection: React.FC = () => {
  const frame = useCurrentFrame();

  const headP = reveal(frame, 2, 18);

  return (
    <AbsoluteFill>
      <Background glow="indigo" />

      {/* header — explicitly future */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 78,
          textAlign: "center",
          fontFamily: FONT,
          opacity: headP,
        }}
      >
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 12,
            padding: "10px 24px",
            borderRadius: 999,
            background: `${COLORS.cyan}16`,
            border: `1px solid ${COLORS.cyan}88`,
            color: COLORS.cyan,
            fontSize: 24,
            fontWeight: 700,
            letterSpacing: 1,
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: 999,
              background: COLORS.cyan,
            }}
          />
          POSSIBLE NEXT STEP · not implemented in Mettle
        </div>
      </div>

      {/* ---- arrows ---- */}
      {/* traces -> diagnose / evals */}
      <FlowArrow
        id="fd-1a"
        from={{ x: 314, y: 500 }}
        to={{ x: 342, y: 448 }}
        progress={reveal(frame, 22, 16)}
        color={COLORS.textFaint}
        width={2.4}
        curve={20}
      />
      <FlowArrow
        id="fd-1b"
        from={{ x: 314, y: 560 }}
        to={{ x: 342, y: 636 }}
        progress={reveal(frame, 30, 16)}
        color={COLORS.textFaint}
        width={2.4}
        curve={20}
      />
      {/* diagnose/evals -> selector */}
      <FlowArrow
        id="fd-2a"
        from={{ x: 598, y: 448 }}
        to={{ x: 636, y: 486 }}
        progress={reveal(frame, 60, 16)}
        color={COLORS.indigo}
        width={2.6}
        curve={20}
      />
      <FlowArrow
        id="fd-2b"
        from={{ x: 598, y: 636 }}
        to={{ x: 636, y: 600 }}
        progress={reveal(frame, 66, 16)}
        color={COLORS.indigo}
        width={2.6}
        curve={20}
      />
      {/* selector -> strategies */}
      <FlowArrow
        id="fd-3a"
        from={{ x: 992, y: 500 }}
        to={{ x: 1026, y: 384 }}
        progress={reveal(frame, 150, 18)}
        color={COLORS.indigo}
        width={2.6}
        curve={30}
      />
      <FlowArrow
        id="fd-3b"
        from={{ x: 992, y: 564 }}
        to={{ x: 1026, y: 546 }}
        progress={reveal(frame, 158, 18)}
        color={COLORS.indigo}
        width={2.6}
      />
      <FlowArrow
        id="fd-3c"
        from={{ x: 992, y: 628 }}
        to={{ x: 1026, y: 708 }}
        progress={reveal(frame, 166, 18)}
        color={COLORS.indigo}
        width={2.6}
        curve={30}
      />
      {/* strategies -> evaluate */}
      <FlowArrow
        id="fd-4a"
        from={{ x: 1330, y: 384 }}
        to={{ x: 1396, y: 452 }}
        progress={reveal(frame, 208, 16)}
        color={COLORS.textDim}
        width={2.4}
        curve={30}
      />
      <FlowArrow
        id="fd-4b"
        from={{ x: 1330, y: 546 }}
        to={{ x: 1396, y: 480 }}
        progress={reveal(frame, 214, 16)}
        color={COLORS.textDim}
        width={2.4}
        curve={30}
      />
      <FlowArrow
        id="fd-4c"
        from={{ x: 1330, y: 708 }}
        to={{ x: 1396, y: 508 }}
        progress={reveal(frame, 220, 16)}
        color={COLORS.textDim}
        width={2.4}
        curve={30}
      />
      {/* evaluate -> select */}
      <FlowArrow
        id="fd-5"
        from={{ x: 1546, y: 588 }}
        to={{ x: 1546, y: 616 }}
        progress={reveal(frame, 250, 14)}
        color={COLORS.green}
        width={2.6}
      />

      {/* ---- upstream (dimmed existing context) ---- */}
      <NodeCard
        n={{
          x: 74,
          y: 438,
          w: 240,
          h: 168,
          title: "Agent traces & failures",
          sub: "tool calls, failed outcomes",
          accent: "muted",
        }}
        frame={frame}
        startFrame={8}
        dim
      />
      <NodeCard
        n={{
          x: 342,
          y: 372,
          w: 256,
          h: 150,
          title: "Diagnose recurring issue",
          accent: "muted",
        }}
        frame={frame}
        startFrame={16}
        dim
      />
      <NodeCard
        n={{
          x: 342,
          y: 560,
          w: 256,
          h: 150,
          title: "Build targeted evals",
          accent: "muted",
        }}
        frame={frame}
        startFrame={24}
        dim
      />

      {/* ---- focal: Repair Strategy Selector ---- */}
      <StrategySelector
        x={636}
        y={352}
        w={356}
        h={424}
        frame={frame}
        startFrame={80}
      />

      {/* ---- strategies ---- */}
      {STRAT.map((s, i) => (
        <NodeCard
          key={s.title}
          n={{
            x: 1026,
            y: s.y,
            w: 304,
            h: s.h,
            title: s.title,
            sub: s.sub,
            tag: s.tag,
            accent: i === 2 ? "muted" : i === 1 ? "cyan" : "indigo",
          }}
          frame={frame}
          startFrame={176 + i * 8}
        />
      ))}

      {/* ---- evaluate + select ---- */}
      <NodeCard
        n={{
          x: 1396,
          y: 372,
          w: 300,
          h: 216,
          title: "Evaluate candidate",
          accent: "cyan",
          checks: ["target gain", "held-out behavior", "regressions", "cost"],
        }}
        frame={frame}
        startFrame={228}
      />
      <NodeCard
        n={{
          x: 1396,
          y: 616,
          w: 300,
          h: 140,
          title: "Select repair",
          sub: "deploy, retry, or reject",
          accent: "indigo",
        }}
        frame={frame}
        startFrame={256}
      />
    </AbsoluteFill>
  );
};
