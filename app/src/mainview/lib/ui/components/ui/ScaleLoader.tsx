import type { ThemeColorName } from "../../utils/getThemeColor";
import { useTheme } from "../../providers/ThemeProvider";
import { getThemeColor } from "../../utils/getThemeColor";

type ScaleLoaderProps = {
  className?: string;
  height?: number;
  width?: number;
  animationColor?: ThemeColorName;
};

export function ScaleLoader({
  animationColor = "foreground",
  className,
  height = 18,
  width = 3,
}: ScaleLoaderProps) {
  const { darkOrLightTheme } = useTheme();
  const color = getThemeColor(darkOrLightTheme, animationColor);

  return (
    <div
      className={`
        flex items-center justify-center space-x-1

        ${className ?? ""}
      `}
    >
      {[0, 1, 2, 3, 4].map((index) => (
        <div
          className="animate-scale"
          key={index}
          style={{
            animationDelay: `${index * 0.1}s`,
            backgroundColor: color,
            borderRadius: "2px",
            display: "inline-block",
            height: `${height}px`,
            margin: "0 2px",
            width: `${width}px`,
          }}
        />
      ))}
      <style>{`
        @keyframes scale {
          0%, 100% {
            transform: scaleY(1.0);
          }
          50% {
            transform: scaleY(1.8);
          }
        }
        .animate-scale {
          animation: scale 1200ms ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
