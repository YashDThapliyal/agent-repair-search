import { AbsoluteFill } from "remotion";
import { COLORS } from "../theme";

// Shared restrained backdrop: deep navy radial base, faint dot grid, and a
// single soft indigo glow. No particles, no motion by default.
export const Background: React.FC<{ glow?: "indigo" | "cyan" | "none" }> = ({
  glow = "indigo",
}) => {
  const glowColor =
    glow === "cyan"
      ? "rgba(84, 199, 220, 0.10)"
      : glow === "none"
        ? "transparent"
        : "rgba(139, 124, 240, 0.12)";

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {/* base vertical depth */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(120% 90% at 50% -10%, ${COLORS.bg} 0%, ${COLORS.bgDeep} 75%)`,
        }}
      />
      {/* faint technical dot grid */}
      <AbsoluteFill
        style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0)`,
          backgroundSize: "48px 48px",
          maskImage:
            "radial-gradient(80% 70% at 50% 45%, black 40%, transparent 100%)",
          WebkitMaskImage:
            "radial-gradient(80% 70% at 50% 45%, black 40%, transparent 100%)",
        }}
      />
      {/* single soft accent glow */}
      {glow !== "none" && (
        <AbsoluteFill
          style={{
            background: `radial-gradient(45% 40% at 50% 42%, ${glowColor} 0%, transparent 70%)`,
          }}
        />
      )}
    </AbsoluteFill>
  );
};
