import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { MetricBar } from "../components/MetricBar";
import { Kicker } from "../components/primitives";
import { StageTag } from "./Failure";
import { COLORS, FONT, RESULTS } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 5 (0:35-0:45): the final result.
// All numbers are the exact composite mean scores from
// reports/final_experiment.json. Bars use a zoomed axis (labeled) so small but
// real differences are legible; the printed number is always the true value.

const PANEL = {
  leftX: 84,
  rightX: 972,
  y: 244,
  w: 864,
  h: 462,
};

export const Results: React.FC = () => {
  const frame = useCurrentFrame();

  const frameP = reveal(frame, 8, 20); // framing line
  const t1 = reveal(frame, 96, 22); // held-out takeaway
  const t2 = reveal(frame, 206, 22); // regression takeaway
  const bottom1 = reveal(frame, 238, 22);
  const bottom2 = reveal(frame, 262, 22);

  return (
    <AbsoluteFill>
      <Background glow="indigo" />
      <StageTag />

      <Kicker
        style={{
          position: "absolute",
          left: 90,
          top: 96,
          opacity: reveal(frame, 4, 16),
        }}
      >
        Final result
      </Kicker>

      {/* ---- Framing: what the two panels measure ---- */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 178,
          textAlign: "center",
          fontFamily: FONT,
          fontSize: 31,
          fontWeight: 600,
          color: COLORS.textDim,
          opacity: frameP,
          translate: `0px ${(1 - frameP) * 12}px`,
        }}
      >
        A good repair must{" "}
        <span style={{ color: COLORS.indigo, fontWeight: 750 }}>
          fix new cases
        </span>{" "}
        — without{" "}
        <span style={{ color: COLORS.green, fontWeight: 750 }}>
          breaking what already worked
        </span>
        .
      </div>

      {/* ---- Held-out panel ---- */}
      <Panel
        x={PANEL.leftX}
        title="Held-out behavior"
        sub="Unseen cases, kept out of repair search — does the fix generalize?"
        scale="axis 0.40 – 0.60"
        frame={frame}
        startFrame={16}
      >
        <MetricBar
          label="Original"
          value={RESULTS.original.heldout}
          frame={frame}
          startFrame={34}
          width={760}
          domain={[0.4, 0.6]}
          color={COLORS.textFaint}
          labelColor={COLORS.textFaint}
        />
        <MetricBar
          label="Focused repair"
          value={RESULTS.focused.heldout}
          frame={frame}
          startFrame={46}
          width={760}
          domain={[0.4, 0.6]}
          color={COLORS.indigo}
          highlight
        />
        <MetricBar
          label="Bounded search (GEPA)"
          value={RESULTS.gepa.heldout}
          frame={frame}
          startFrame={58}
          width={760}
          domain={[0.4, 0.6]}
          color={COLORS.cyan}
        />
      </Panel>
      <Caption
        x={PANEL.leftX}
        y={724}
        w={PANEL.w}
        text="Focused repair generalized better"
        color={COLORS.indigo}
        p={t1}
      />

      {/* ---- Regression panel ---- */}
      <Panel
        x={PANEL.rightX}
        title="Regression"
        sub="Unrelated behavior that should still pass — did the fix break it?"
        scale="axis 0.90 – 0.97"
        frame={frame}
        startFrame={120}
      >
        <MetricBar
          label="Original"
          value={RESULTS.original.regression}
          frame={frame}
          startFrame={140}
          width={760}
          domain={[0.9, 0.97]}
          color={COLORS.textFaint}
          labelColor={COLORS.textFaint}
        />
        <MetricBar
          label="Focused repair"
          value={RESULTS.focused.regression}
          frame={frame}
          startFrame={152}
          width={760}
          domain={[0.9, 0.97]}
          color={COLORS.indigo}
        />
        <MetricBar
          label="Bounded search (GEPA)"
          value={RESULTS.gepa.regression}
          frame={frame}
          startFrame={164}
          width={760}
          domain={[0.9, 0.97]}
          color={COLORS.green}
          highlight
        />
      </Panel>
      <Caption
        x={PANEL.rightX}
        y={724}
        w={PANEL.w}
        text="GEPA had the strongest regression score"
        color={COLORS.green}
        p={t2}
      />

      {/* ---- Takeaways ---- */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 812,
          textAlign: "center",
          fontFamily: FONT,
        }}
      >
        <div
          style={{
            fontSize: 46,
            fontWeight: 750,
            color: COLORS.text,
            ...riseStyle(bottom1, 20),
          }}
        >
          Different repair strategies produced different tradeoffs.
        </div>
        <div
          style={{
            fontSize: 32,
            fontWeight: 600,
            color: COLORS.indigo,
            marginTop: 16,
            opacity: bottom2,
          }}
        >
          More search was not universally better.
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Panel: React.FC<{
  x: number;
  title: string;
  sub: string;
  scale: string;
  frame: number;
  startFrame: number;
  children: React.ReactNode;
}> = ({ x, title, sub, scale, frame, startFrame, children }) => {
  const p = reveal(frame, startFrame, 22);
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: PANEL.y,
        width: PANEL.w,
        height: PANEL.h,
        borderRadius: 22,
        boxSizing: "border-box",
        padding: "26px 44px 30px",
        background:
          "linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015))",
        border: `1.5px solid ${COLORS.border}`,
        boxShadow: "0 18px 50px rgba(0,0,0,0.3)",
        fontFamily: FONT,
        opacity: p,
        translate: `0px ${(1 - p) * 24}px`,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          minHeight: 96,
        }}
      >
        <div style={{ maxWidth: 560 }}>
          <div style={{ fontSize: 34, fontWeight: 750, color: COLORS.text }}>
            {title}
          </div>
          <div
            style={{
              fontSize: 20,
              color: COLORS.textDim,
              marginTop: 6,
              lineHeight: 1.35,
            }}
          >
            {sub}
          </div>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          <div
            style={{ fontSize: 18, color: COLORS.textFaint, fontWeight: 500 }}
          >
            {scale}
          </div>
          <div
            style={{
              fontSize: 17,
              color: COLORS.green,
              fontWeight: 600,
              marginTop: 6,
            }}
          >
            ↑ higher is better
          </div>
        </div>
      </div>
      <div
        style={{
          marginTop: 26,
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        {children}
      </div>
    </div>
  );
};

const Caption: React.FC<{
  x: number;
  y: number;
  w: number;
  text: string;
  color: string;
  p: number;
}> = ({ x, y, w, text, color, p }) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: w,
      textAlign: "center",
      fontFamily: FONT,
      fontSize: 30,
      fontWeight: 700,
      color,
      opacity: p,
      translate: `0px ${(1 - p) * 14}px`,
    }}
  >
    {text}
  </div>
);
