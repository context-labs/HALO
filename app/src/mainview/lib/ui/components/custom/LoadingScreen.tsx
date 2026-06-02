import { Centered } from "~/lib/ui/components/custom/Centered";
import { ScaleLoader } from "~/lib/ui/components/ui/ScaleLoader";

export function LoadingScreen() {
  return (
    <Centered className="absolute left-0 top-0 h-full w-full bg-background">
      <ScaleLoader />
    </Centered>
  );
}
