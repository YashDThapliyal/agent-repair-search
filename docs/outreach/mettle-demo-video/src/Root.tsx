import "./index.css";
import { Composition } from "remotion";
import { MettleDemo } from "./MettleDemo";
import { FPS, TOTAL_FRAMES } from "./theme";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="MettleDemo"
        component={MettleDemo}
        durationInFrames={TOTAL_FRAMES}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
