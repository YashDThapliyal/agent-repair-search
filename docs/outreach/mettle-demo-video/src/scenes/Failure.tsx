import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { FlowArrow } from "../components/FlowArrow";
import { StatusGlyph } from "../components/AgentTrace";
import { Kicker } from "../components/primitives";
import { COLORS, FONT, MONO } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 2 (0:06-0:14): the failure being studied.
// Left: the abstract stateful tool-calling pipeline (context — where the
// failure happens). Right: one concrete, faithful case. For a "money-back"
// request on a still-pending charge, the frozen policy precedence requires
// reverse_pending_charge (rank 4), but an agent routing on the words alone
// jumps to issue_refund (rank 7). Grounded in scenario.json:
//   diagnosis: "selects tools from surface wording alone instead of the frozen
//   policy precedence applied to the provided case state."
//   policy_precedence: [... reverse_pending_charge(4) ... issue_refund(7) ...]

const FLOW = [
  { label: "User request", accent: "muted" as const },
  { label: "Account state", accent: "amber" as const },
  { label: "Tool call", accent: "muted" as const },
  { label: "State changes", accent: "green" as const },
  { label: "Next routing decision", accent: "indigo" as const },
];

const CX = 300;
const NODE_W = 300;
const YS = [230, 355, 480, 605, 730];

const PANEL = { x: 640, y: 190, w: 1190, h: 610 };

export const Failure: React.FC = () => {
  const frame = useCurrentFrame();

  const panelP = reveal(frame, 28, 22);
  const quoteP = reveal(frame, 50, 20);
  const stateP = reveal(frame, 80, 20);
  const agentP = reveal(frame, 108, 22);
  const correctP = reveal(frame, 140, 22);
  const noteP = reveal(frame, 172, 20);
  const labelP = reveal(frame, 198, 20);

  const accentOf = (a: string) =>
    a === "amber"
      ? COLORS.amber
      : a === "green"
        ? COLORS.green
        : a === "indigo"
          ? COLORS.indigo
          : COLORS.borderStrong;

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
        The failure being studied
      </Kicker>

      {/* ---- Left: abstract stateful pipeline (context) ---- */}
      {FLOW.slice(1).map((_, i) => (
        <FlowArrow
          key={i}
          id={`f${i}`}
          from={{ x: CX, y: YS[i] + 48 }}
          to={{ x: CX, y: YS[i + 1] - 48 }}
          progress={reveal(frame, 12 + i * 12, 12)}
          color={COLORS.textFaint}
          width={2.5}
        />
      ))}
      {FLOW.map((n, i) => {
        const p = reveal(frame, 8 + i * 12, 16);
        const ac = accentOf(n.accent);
        const isAccent = n.accent !== "muted";
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: CX - NODE_W / 2,
              top: YS[i] - 48,
              width: NODE_W,
              height: 96,
              borderRadius: 16,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              textAlign: "center",
              padding: "0 18px",
              boxSizing: "border-box",
              background: isAccent ? `${ac}14` : "rgba(255,255,255,0.03)",
              border: `1.5px solid ${isAccent ? `${ac}99` : COLORS.border}`,
              color: isAccent ? COLORS.text : COLORS.textDim,
              fontFamily: FONT,
              fontSize: 27,
              fontWeight: 650,
              ...riseStyle(p, 18),
            }}
          >
            {n.label}
          </div>
        );
      })}

      {/* ---- Right: one concrete, faithful case ---- */}
      <div
        style={{
          position: "absolute",
          left: PANEL.x,
          top: PANEL.y,
          width: PANEL.w,
          height: PANEL.h,
          borderRadius: 22,
          boxSizing: "border-box",
          padding: "30px 44px",
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015))",
          border: `1.5px solid ${COLORS.border}`,
          boxShadow: "0 18px 50px rgba(0,0,0,0.3)",
          fontFamily: FONT,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 24,
          opacity: panelP,
          translate: `${(1 - panelP) * 26}px 0px`,
        }}
      >
        {/* the customer request */}
        <div style={{ opacity: quoteP }}>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 18,
              letterSpacing: 1.5,
              color: COLORS.textFaint,
              textTransform: "uppercase",
            }}
          >
            Customer says
          </div>
          <div
            style={{
              fontSize: 40,
              fontWeight: 700,
              color: COLORS.text,
              marginTop: 8,
            }}
          >
            &ldquo;I want my money back.&rdquo;
          </div>
        </div>

        {/* the account state that should decide routing */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            opacity: stateP,
          }}
        >
          <span style={{ fontSize: 21, color: COLORS.textDim }}>
            Account state:
          </span>
          <StateChip text="charge still pending" color={COLORS.amber} />
          <StateChip text="identity verified" color={COLORS.green} />
        </div>

        <div
          style={{ height: 1, background: COLORS.border, opacity: stateP }}
        />

        {/* the contrast: agent's wording pick vs. the policy+state answer */}
        <div style={{ display: "flex", alignItems: "stretch", gap: 20 }}>
          <ToolCard
            heading="The agent routes on the words"
            tool="issue_refund"
            rankLabel="policy rank 7"
            ok={false}
            p={agentP}
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              fontSize: 34,
              fontWeight: 800,
              color: COLORS.textFaint,
              opacity: correctP,
            }}
          >
            &ne;
          </div>
          <ToolCard
            heading="State + policy require"
            tool="reverse_pending_charge"
            rankLabel="policy rank 4"
            ok
            p={correctP}
          />
        </div>

        {/* why: the precedence rule, in plain terms */}
        <div
          style={{
            fontSize: 22,
            fontWeight: 500,
            color: COLORS.textDim,
            lineHeight: 1.4,
            opacity: noteP,
          }}
        >
          A pending charge must be{" "}
          <span style={{ color: COLORS.green, fontWeight: 700 }}>reversed</span>{" "}
          before a refund — so the agent picks a lower-precedence tool and
          misroutes.
        </div>
      </div>

      {/* ---- Failure label ---- */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 838,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          fontFamily: FONT,
          opacity: labelP,
          translate: `0px ${(1 - labelP) * 16}px`,
        }}
      >
        <div style={{ fontSize: 44, fontWeight: 750, color: COLORS.text }}>
          State-dependent tool routing failure
        </div>
        <div
          style={{
            fontSize: 27,
            fontWeight: 500,
            color: COLORS.textDim,
            marginTop: 12,
          }}
        >
          The right tool depends on account state and policy order, not the
          user&rsquo;s words.
        </div>
      </div>
    </AbsoluteFill>
  );
};

const StateChip: React.FC<{ text: string; color: string }> = ({
  text,
  color,
}) => (
  <span
    style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      padding: "9px 18px",
      borderRadius: 999,
      fontFamily: MONO,
      fontSize: 21,
      fontWeight: 600,
      color,
      background: `${color}14`,
      border: `1px solid ${color}88`,
      whiteSpace: "nowrap",
    }}
  >
    <span
      style={{
        width: 9,
        height: 9,
        borderRadius: 999,
        background: color,
      }}
    />
    {text}
  </span>
);

const ToolCard: React.FC<{
  heading: string;
  tool: string;
  rankLabel: string;
  ok: boolean;
  p: number;
}> = ({ heading, tool, rankLabel, ok, p }) => {
  const color = ok ? COLORS.green : COLORS.red;
  return (
    <div
      style={{
        flex: 1,
        borderRadius: 16,
        padding: "20px 24px",
        boxSizing: "border-box",
        background: `${color}12`,
        border: `1.6px solid ${color}`,
        boxShadow: `0 0 26px ${color}1F`,
        display: "flex",
        flexDirection: "column",
        gap: 12,
        opacity: p,
        translate: `0px ${(1 - p) * 16}px`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <StatusGlyph ok={ok} size={30} />
        <span
          style={{
            fontSize: 21,
            fontWeight: 650,
            color: COLORS.text,
          }}
        >
          {heading}
        </span>
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 30,
          fontWeight: 700,
          color,
        }}
      >
        {tool}
      </div>
      <span
        style={{
          fontFamily: MONO,
          fontSize: 17,
          color: COLORS.textFaint,
        }}
      >
        {rankLabel}
      </span>
    </div>
  );
};

// A small persistent tag distinguishing built work from future direction.
export const StageTag: React.FC<{ future?: boolean }> = ({ future }) => {
  const frame = useCurrentFrame();
  const p = reveal(frame, 2, 16);
  const color = future ? COLORS.cyan : COLORS.green;
  return (
    <div
      style={{
        position: "absolute",
        right: 90,
        top: 96,
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        padding: "9px 18px",
        borderRadius: 999,
        background: `${color}16`,
        border: `1px solid ${color}88`,
        color,
        fontFamily: FONT,
        fontSize: 22,
        fontWeight: 650,
        opacity: p,
      }}
    >
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 999,
          background: color,
        }}
      />
      {future ? "Possible next step" : "Completed experiment"}
    </div>
  );
};
