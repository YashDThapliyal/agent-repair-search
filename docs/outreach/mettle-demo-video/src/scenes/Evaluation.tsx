import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { FlowArrow } from "../components/FlowArrow";
import { EvalStageCard, Badge, GateStage } from "../components/EvalGate";
import { Kicker } from "../components/primitives";
import { StageTag } from "./Failure";
import { COLORS, FONT } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 4 (0:25-0:35): the evaluation protocol.
// Every repair candidate (Original / Focused / Bounded search) runs the same
// three-stage gate. The held-out split is loaded only after candidates are
// frozen and is never reused — grounded in reports/final_experiment.json
// (heldout_pristine=true, heldout_reused=false).

const ARMS = [
  // Hex colors only — chip bg/border use ${color}AA hex-alpha concatenation.
  { label: "Original", color: "#6B7488" },
  { label: "Focused repair", color: COLORS.indigo },
  { label: "Bounded search", color: COLORS.cyan },
];

const STAGES: GateStage[] = [
  {
    index: "1",
    title: "Optimize",
    sub: "improve target behavior",
    accent: "indigo",
  },
  {
    index: "2",
    title: "Untouched held-out",
    sub: "does the repair generalize?",
    accent: "cyan",
    locked: true,
  },
  {
    index: "3",
    title: "Regression",
    sub: "did existing behavior break?",
    accent: "green",
  },
];

const STAGE_X = [556, 1010, 1464];
const STAGE_W = 386;
const STAGE_Y = 336;
const STAGE_H = 288;

export const Evaluation: React.FC = () => {
  const frame = useCurrentFrame();

  const armY = [430, 512, 594];

  // Traveling candidate marker across the pipeline (after cards are in).
  const travel = interpolate(frame, [190, 285], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const markerX = interpolate(
    travel,
    [0, 1],
    [STAGE_X[0] + 40, STAGE_X[2] + STAGE_W - 40],
  );

  return (
    <AbsoluteFill>
      <Background glow="cyan" />
      <StageTag />

      <Kicker
        style={{
          position: "absolute",
          left: 90,
          top: 96,
          opacity: reveal(frame, 4, 16),
        }}
      >
        Evaluation protocol
      </Kicker>

      {/* arms -> first stage */}
      {armY.map((y, i) => (
        <FlowArrow
          key={i}
          id={`ev-arm${i}`}
          from={{ x: 402, y }}
          to={{ x: STAGE_X[0] - 4, y: STAGE_Y + STAGE_H / 2 }}
          progress={reveal(frame, 40 + i * 6, 20)}
          color={ARMS[i].color}
          width={2.6}
          curve={40}
        />
      ))}

      {/* arm chips */}
      {ARMS.map((a, i) => {
        const p = reveal(frame, 8 + i * 8, 16);
        return (
          <div
            key={a.label}
            style={{
              position: "absolute",
              left: 90,
              top: armY[i] - 34,
              width: 300,
              height: 68,
              borderRadius: 14,
              display: "flex",
              alignItems: "center",
              paddingLeft: 22,
              boxSizing: "border-box",
              background: `${a.color}12`,
              border: `1.5px solid ${a.color}88`,
              fontFamily: FONT,
              fontSize: 26,
              fontWeight: 700,
              color: COLORS.text,
              ...riseStyle(p, 16),
            }}
          >
            {a.label}
          </div>
        );
      })}

      {/* arrows between stages */}
      <FlowArrow
        id="ev-s1"
        from={{ x: STAGE_X[0] + STAGE_W, y: STAGE_Y + STAGE_H / 2 }}
        to={{ x: STAGE_X[1], y: STAGE_Y + STAGE_H / 2 }}
        progress={reveal(frame, 110, 16)}
        color={COLORS.textDim}
        width={3}
      />
      <FlowArrow
        id="ev-s2"
        from={{ x: STAGE_X[1] + STAGE_W, y: STAGE_Y + STAGE_H / 2 }}
        to={{ x: STAGE_X[2], y: STAGE_Y + STAGE_H / 2 }}
        progress={reveal(frame, 150, 16)}
        color={COLORS.textDim}
        width={3}
      />

      {/* stage cards */}
      {STAGES.map((s, i) => (
        <div
          key={s.index}
          style={{ position: "absolute", left: STAGE_X[i], top: STAGE_Y }}
        >
          <EvalStageCard
            stage={s}
            frame={frame}
            startFrame={70 + i * 40}
            w={STAGE_W}
            h={STAGE_H}
          />
        </div>
      ))}

      {/* traveling candidate marker */}
      {travel > 0 && travel < 1 && (
        <div
          style={{
            position: "absolute",
            left: markerX - 9,
            top: STAGE_Y + STAGE_H / 2 - 9,
            width: 18,
            height: 18,
            borderRadius: 999,
            background: COLORS.text,
            boxShadow: `0 0 20px ${COLORS.cyan}`,
          }}
        />
      )}

      {/* badge */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 720,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <Badge frame={frame} startFrame={170} color={COLORS.cyan}>
          Held-out never reused
        </Badge>
      </div>

      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 800,
          textAlign: "center",
          fontFamily: FONT,
          fontSize: 26,
          fontWeight: 500,
          color: COLORS.textDim,
          opacity: reveal(frame, 196, 20),
        }}
      >
        Held-out is loaded only after every candidate is frozen.
      </div>
    </AbsoluteFill>
  );
};
