import { interpolate } from "remotion";
import { COLORS } from "../theme";

// A progressively-drawn connector in the composition's native 1920x1080 space.
// Renders its own transparent full-frame SVG so scenes can place HTML cards at
// matching pixel coordinates and let arrows link them behind the content.
//
// `progress` (0..1) drives both the line draw and the arrowhead fade-in.

type Point = { x: number; y: number };

export const FlowArrow: React.FC<{
  from: Point;
  to: Point;
  progress: number;
  color?: string;
  width?: number;
  // Optional curvature: horizontal S-curve amount in px. 0 = straight.
  curve?: number;
  dashed?: boolean;
  head?: boolean;
  id: string;
}> = ({
  from,
  to,
  progress,
  color = COLORS.textFaint,
  width = 3,
  curve = 0,
  dashed = false,
  head = true,
  id,
}) => {
  const d =
    curve === 0
      ? `M ${from.x} ${from.y} L ${to.x} ${to.y}`
      : `M ${from.x} ${from.y} C ${from.x + curve} ${from.y}, ${
          to.x - curve
        } ${to.y}, ${to.x} ${to.y}`;

  const drawn = interpolate(progress, [0, 0.85], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const headOpacity = interpolate(progress, [0.7, 1], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <svg
      width={1920}
      height={1080}
      viewBox="0 0 1920 1080"
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    >
      {head && (
        <defs>
          <marker
            id={`ah-${id}`}
            markerWidth="9"
            markerHeight="9"
            refX="6.5"
            refY="4.5"
            orient="auto"
          >
            <path
              d="M 0 0 L 9 4.5 L 0 9 z"
              fill={color}
              opacity={headOpacity}
            />
          </marker>
        </defs>
      )}
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={width}
        strokeLinecap="round"
        pathLength={1}
        strokeDasharray={dashed ? "0.02 0.03" : 1}
        strokeDashoffset={dashed ? 0 : 1 - drawn}
        opacity={dashed ? drawn * 0.9 : 1}
        markerEnd={head ? `url(#ah-${id})` : undefined}
      />
    </svg>
  );
};
