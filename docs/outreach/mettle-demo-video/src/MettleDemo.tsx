import {
  AbsoluteFill,
  interpolate,
  Series,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { SCENES } from "./theme";
import { Opening } from "./scenes/Opening";
import { Failure } from "./scenes/Failure";
import { RepairComparison } from "./scenes/RepairComparison";
import { Evaluation } from "./scenes/Evaluation";
import { Results } from "./scenes/Results";
import { FutureDirection } from "./scenes/FutureDirection";
import { Connection } from "./scenes/Connection";
import { Closing } from "./scenes/Closing";
import { COLORS } from "./theme";

// Softens the hard cuts between scenes with a short fade in (and out, except
// the final scene which should hold on its last frame). Fades are brief so the
// dip-to-black reads as intentional pacing, not dead time.
const SceneFade: React.FC<{
  fadeOut?: boolean;
  children: React.ReactNode;
}> = ({ fadeOut = true, children }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const fadeIn = interpolate(frame, [0, 7], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const out = fadeOut
    ? interpolate(frame, [durationInFrames - 7, durationInFrames], [1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      })
    : 1;
  return (
    <AbsoluteFill style={{ opacity: Math.min(fadeIn, out) }}>
      {children}
    </AbsoluteFill>
  );
};

// Top-level composition: eight scenes played back to back via Series so each
// scene's useCurrentFrame() is local (starts at 0). Durations live in theme.ts.
export const MettleDemo: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      <Series>
        <Series.Sequence durationInFrames={SCENES.opening}>
          <SceneFade>
            <Opening />
          </SceneFade>
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENES.failure}>
          <SceneFade>
            <Failure />
          </SceneFade>
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENES.repair}>
          <SceneFade>
            <RepairComparison />
          </SceneFade>
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENES.evaluation}>
          <SceneFade>
            <Evaluation />
          </SceneFade>
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENES.results}>
          <SceneFade>
            <Results />
          </SceneFade>
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENES.future}>
          <SceneFade>
            <FutureDirection />
          </SceneFade>
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENES.connection}>
          <SceneFade>
            <Connection />
          </SceneFade>
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENES.closing}>
          <SceneFade fadeOut={false}>
            <Closing />
          </SceneFade>
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
