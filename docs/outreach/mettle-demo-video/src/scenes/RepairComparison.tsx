import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { FlowArrow } from "../components/FlowArrow";
import { Kicker } from "../components/primitives";
import { StageTag } from "./Failure";
import { COLORS, FONT, MONO } from "../theme";
import { reveal, riseStyle } from "../util";

// Scene 3 (0:14-0:25): what Mettle actually compares.
// A failing agent branches into three arms: Original (muted reference),
// Focused repair (one targeted proposal), Bounded search (search over
// candidates, e.g. GEPA). Funnel numbers are grounded in search accounting:
// 10 proposals -> distinct candidates -> 1 selected.

// A schematic "patch": colored bars standing in for added / edited lines over
// the editable artifacts (system prompt + tool descriptions). Deliberately not
// literal patch text — an abstract representation of a change.
const PatchMini: React.FC<{
  rows: ("add" | "edit" | "ctx")[];
  frame: number;
  startFrame: number;
  compact?: boolean;
}> = ({ rows, frame, startFrame, compact }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: compact ? 6 : 9 }}>
    {rows.map((r, i) => {
      const p = reveal(frame, startFrame + i * 4, 10);
      const color =
        r === "add" ? COLORS.green : r === "edit" ? COLORS.cyan : COLORS.textFaint;
      const w = r === "ctx" ? 0.45 : r === "add" ? 0.9 : 0.68;
      return (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            opacity: p,
            translate: `${(1 - p) * -10}px 0px`,
          }}
        >
          <span
            style={{
              fontFamily: MONO,
              fontSize: compact ? 14 : 18,
              color,
              width: 14,
            }}
          >
            {r === "add" ? "+" : r === "edit" ? "~" : ""}
          </span>
          <span
            style={{
              height: compact ? 7 : 10,
              width: `${w * 100}%`,
              borderRadius: 999,
              background: r === "ctx" ? "rgba(255,255,255,0.10)" : `${color}99`,
            }}
          />
        </div>
      );
    })}
  </div>
);

export const RepairComparison: React.FC = () => {
  const frame = useCurrentFrame();

  const srcP = reveal(frame, 6, 20);
  const focusFrameP = reveal(frame, 55, 20);
  const origP = reveal(frame, 62, 20);
  const searchP = reveal(frame, 92, 20);

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
        What Mettle compares
      </Kicker>

      {/* ---- Branch arrows (behind cards) ---- */}
      <FlowArrow
        id="rc-top"
        from={{ x: 402, y: 470 }}
        to={{ x: 556, y: 310 }}
        progress={reveal(frame, 30, 20)}
        color={COLORS.indigo}
        width={3}
        curve={80}
      />
      <FlowArrow
        id="rc-mid"
        from={{ x: 402, y: 540 }}
        to={{ x: 556, y: 540 }}
        progress={reveal(frame, 40, 18)}
        color={COLORS.textFaint}
        width={3}
      />
      <FlowArrow
        id="rc-bot"
        from={{ x: 402, y: 610 }}
        to={{ x: 556, y: 760 }}
        progress={reveal(frame, 50, 20)}
        color={COLORS.cyan}
        width={3}
        curve={80}
      />

      {/* ---- Source: failing agent ---- */}
      <div
        style={{
          position: "absolute",
          left: 90,
          top: 448,
          width: 312,
          height: 184,
          borderRadius: 18,
          boxSizing: "border-box",
          padding: "24px 26px",
          background: `${COLORS.red}12`,
          border: `1.5px solid ${COLORS.red}88`,
          boxShadow: `0 0 34px ${COLORS.red}22`,
          fontFamily: FONT,
          ...riseStyle(srcP, 22),
        }}
      >
        <div style={{ fontSize: 30, fontWeight: 750, color: COLORS.text }}>
          Failing agent
        </div>
        <div
          style={{
            fontSize: 20,
            color: COLORS.textDim,
            marginTop: 10,
            lineHeight: 1.4,
          }}
        >
          diagnosed failure +
          <br />
          editable artifacts
        </div>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 16,
            color: COLORS.textFaint,
            marginTop: 12,
          }}
        >
          system prompt · tool descriptions
        </div>
      </div>

      {/* ---- Top branch: Focused repair ---- */}
      <div
        style={{
          position: "absolute",
          left: 556,
          top: 178,
          width: 640,
          height: 264,
          borderRadius: 18,
          boxSizing: "border-box",
          padding: "22px 28px",
          background:
            "linear-gradient(180deg, rgba(139,124,240,0.10), rgba(139,124,240,0.03))",
          border: `1.5px solid ${COLORS.indigo}AA`,
          boxShadow: `0 0 34px ${COLORS.indigo}22, 0 14px 34px rgba(0,0,0,0.3)`,
          fontFamily: FONT,
          ...riseStyle(focusFrameP, 22),
        }}
      >
        <BranchHeader
          title="Focused repair"
          sub="One strong targeted proposal"
          tag="single proposal"
          color={COLORS.indigo}
        />
        <div style={{ marginTop: 18, display: "flex", gap: 30 }}>
          <div style={{ flex: 1 }}>
            <PatchMini
              rows={["ctx", "add", "add", "edit", "edit"]}
              frame={frame}
              startFrame={70}
            />
          </div>
          <div
            style={{
              width: 200,
              fontSize: 19,
              color: COLORS.textDim,
              lineHeight: 1.4,
              alignSelf: "center",
            }}
          >
            one patch, then frozen and evaluated
          </div>
        </div>
      </div>

      {/* ---- Middle branch: Original (muted reference) ---- */}
      <div
        style={{
          position: "absolute",
          left: 556,
          top: 476,
          width: 470,
          height: 128,
          borderRadius: 16,
          boxSizing: "border-box",
          padding: "20px 26px",
          background: "rgba(255,255,255,0.02)",
          border: `1.5px solid ${COLORS.border}`,
          fontFamily: FONT,
          opacity: origP * 0.92,
          translate: `0px ${(1 - origP) * 20}px`,
        }}
      >
        <div style={{ fontSize: 28, fontWeight: 700, color: COLORS.textDim }}>
          Original
        </div>
        <div style={{ fontSize: 20, color: COLORS.textFaint, marginTop: 8 }}>
          unchanged system prompt &amp; tool descriptions
        </div>
      </div>

      {/* ---- Bottom branch: Bounded search ---- */}
      <div
        style={{
          position: "absolute",
          left: 556,
          top: 640,
          width: 1274,
          height: 300,
          borderRadius: 18,
          boxSizing: "border-box",
          padding: "22px 28px",
          background:
            "linear-gradient(180deg, rgba(84,199,220,0.09), rgba(84,199,220,0.025))",
          border: `1.5px solid ${COLORS.cyan}99`,
          boxShadow: `0 0 34px ${COLORS.cyan}1F, 0 14px 34px rgba(0,0,0,0.3)`,
          fontFamily: FONT,
          ...riseStyle(searchP, 22),
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
          }}
        >
          <BranchHeader
            title="Bounded search"
            sub="Search over candidate repairs"
            tag="GEPA"
            color={COLORS.cyan}
          />
          <div
            style={{
              fontFamily: MONO,
              fontSize: 19,
              color: COLORS.textDim,
              textAlign: "right",
              marginTop: 4,
            }}
          >
            {funnelText(frame)}
          </div>
        </div>

        {/* candidate row */}
        <div style={{ marginTop: 20, display: "flex", gap: 16 }}>
          {[0, 1, 2, 3, 4].map((i) => {
            const cp = reveal(frame, 120 + i * 16, 14);
            const selected = i === 3;
            const selP = selected ? reveal(frame, 216, 20) : 0;
            return (
              <div
                key={i}
                style={{
                  flex: 1,
                  height: 150,
                  borderRadius: 12,
                  boxSizing: "border-box",
                  padding: "14px 14px",
                  background: selected
                    ? `${COLORS.cyan}1C`
                    : "rgba(255,255,255,0.03)",
                  border: `1.5px solid ${
                    selected
                      ? `rgba(84,199,220,${0.4 + selP * 0.6})`
                      : COLORS.border
                  }`,
                  boxShadow: selected
                    ? `0 0 ${18 * selP}px ${COLORS.cyan}66`
                    : "none",
                  opacity: cp,
                  translate: `0px ${(1 - cp) * 16}px`,
                  position: "relative",
                }}
              >
                <div
                  style={{
                    fontFamily: MONO,
                    fontSize: 15,
                    color: selected ? COLORS.cyan : COLORS.textFaint,
                    marginBottom: 10,
                  }}
                >
                  cand {i + 1}
                </div>
                <PatchMini
                  rows={
                    [
                      ["ctx", "add", "edit"],
                      ["add", "ctx", "edit"],
                      ["ctx", "edit", "add"],
                      ["add", "edit", "add"],
                      ["ctx", "add", "ctx"],
                    ][i] as ("add" | "edit" | "ctx")[]
                  }
                  frame={frame}
                  startFrame={124 + i * 16}
                  compact
                />
                {selected && selP > 0.15 && (
                  <div
                    style={{
                      position: "absolute",
                      top: 12,
                      right: 12,
                      fontFamily: FONT,
                      fontSize: 15,
                      fontWeight: 700,
                      color: COLORS.cyan,
                      opacity: selP,
                    }}
                  >
                    selected
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Funnel text reveals in stages, grounded in search_accounting.
const funnelText = (frame: number) => {
  if (frame < 118) return "";
  if (frame < 210) return "10 proposals";
  return "10 proposals → candidates → 1 selected";
};

const BranchHeader: React.FC<{
  title: string;
  sub: string;
  tag: string;
  color: string;
}> = ({ title, sub, tag, color }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
    <div>
      <div style={{ fontSize: 32, fontWeight: 750, color: COLORS.text }}>
        {title}
      </div>
      <div style={{ fontSize: 21, color: COLORS.textDim, marginTop: 6 }}>
        {sub}
      </div>
    </div>
    <span
      style={{
        fontFamily: MONO,
        fontSize: 18,
        fontWeight: 700,
        letterSpacing: 1,
        color,
        padding: "6px 14px",
        borderRadius: 8,
        background: `${color}1F`,
        border: `1px solid ${color}66`,
        alignSelf: "flex-start",
      }}
    >
      {tag}
    </span>
  </div>
);
