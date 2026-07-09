import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { AgentTrace, TraceNode } from "../components/AgentTrace";
import { COLORS, FONT } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 1 (0:00-0:06): establish the question.
// An agent trajectory runs, a routing step goes stale and lands on the wrong
// action. Then: "An agent fails." / "What should repair it?"
const TRACE: TraceNode[] = [
  { label: "user request", status: "ok" },
  { label: "verify_identity", status: "ok" },
  { label: "lookup_charge", status: "ok" },
  { label: "route", status: "stale" },
  { label: "wrong action", status: "fail" },
];

export const Opening: React.FC = () => {
  const frame = useCurrentFrame();

  const failP = reveal(frame, 82, 20);
  const askP = reveal(frame, 118, 18);
  const subP = reveal(frame, 146, 18);

  return (
    <AbsoluteFill>
      <Background glow="indigo" />

      <AgentTrace
        nodes={TRACE}
        frame={frame}
        startFrame={6}
        stepFrames={15}
        cx={960}
        cy={330}
        gap={205}
      />

      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 470,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          fontFamily: FONT,
        }}
      >
        <div
          style={{
            fontSize: 116,
            fontWeight: 800,
            color: COLORS.text,
            letterSpacing: -1,
            ...riseStyle(failP, 30),
          }}
        >
          An agent fails.
        </div>

        <div
          style={{
            fontSize: 66,
            fontWeight: 600,
            color: COLORS.indigo,
            marginTop: 26,
            ...riseStyle(askP, 26),
          }}
        >
          What should repair it?
        </div>

        <div
          style={{
            fontSize: 34,
            fontWeight: 500,
            color: COLORS.textDim,
            marginTop: 34,
            opacity: subP,
          }}
        >
          More search is not always the answer.
        </div>
      </div>
    </AbsoluteFill>
  );
};
