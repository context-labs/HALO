import { useId } from "react";

const hashCode = (str: string) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
};

const seededRandom = (seed: number) => {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
};

const generateRandomValues = (seed: number, count: number) => {
  const values: number[] = [];
  for (let i = 0; i < count; i++) {
    values.push(seededRandom(seed + i));
  }
  return values;
};

const getIndex = (randoms: number[], index: number): number => {
  return randoms[index % randoms.length] ?? 0;
};

type PatternType = "circles" | "triangles" | "hexagons";

const PATTERN_RANDOM_VALUE_COUNT = 32;

const assertUnreachable = (value: never): never => {
  throw new Error(`Unexpected pattern type: ${String(value)}`);
};

type PatternAvatarFallbackProps = {
  className?: string;
  name: string;
  showInitials?: boolean;
  size?: number;
  patternType?: PatternType;
};

// Authored by Claude 4 Sonnet.
export function PatternAvatarFallback({
  className = "",
  name,
  patternType = "circles",
  showInitials = true,
  size = 48,
}: PatternAvatarFallbackProps) {
  const gradientId = useId();
  const randoms = generateRandomValues(
    hashCode(name),
    PATTERN_RANDOM_VALUE_COUNT,
  );
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase() ?? "")
    .join("");

  const hue1 = Math.floor(getIndex(randoms, 0) * 360);
  const hue2 = (hue1 + 60 + Math.floor(getIndex(randoms, 1) * 120)) % 360;
  const hue3 = (hue1 + 180 + Math.floor(getIndex(randoms, 2) * 60)) % 360;

  const saturation = 60 + Math.floor(getIndex(randoms, 3) * 30);
  const lightness1 = 45 + Math.floor(getIndex(randoms, 4) * 20);
  const lightness2 = 55 + Math.floor(getIndex(randoms, 5) * 25);
  const lightness3 = 35 + Math.floor(getIndex(randoms, 6) * 30);

  const color1 = `hsl(${hue1}, ${saturation}%, ${lightness1}%)`;
  const color2 = `hsl(${hue2}, ${saturation}%, ${lightness2}%)`;
  const color3 = `hsl(${hue3}, ${saturation}%, ${lightness3}%)`;

  const generatePattern = () => {
    switch (patternType) {
      case "circles":
        return (
          <svg
            className="absolute inset-0"
            height={size}
            width={size}
            viewBox={`0 0 ${size} ${size}`}
            preserveAspectRatio="none"
          >
            <defs>
              <radialGradient cx="50%" cy="50%" id={gradientId} r="50%">
                <stop offset="0%" stopColor={color1} stopOpacity="0.8" />
                <stop offset="70%" stopColor={color2} stopOpacity="0.6" />
                <stop offset="100%" stopColor={color3} stopOpacity="0.9" />
              </radialGradient>
            </defs>
            <rect fill={`url(#${gradientId})`} height="100%" width="100%" />
            {Array.from({ length: 6 }).map((_, i) => (
              <circle
                cx={size * (0.2 + getIndex(randoms, 8 + i) * 0.6)}
                cy={size * (0.2 + getIndex(randoms, 14 + i) * 0.6)}
                fill={i % 2 === 0 ? color2 : color3}
                key={i}
                opacity={0.3 + getIndex(randoms, 10 + i) * 0.4}
                r={size * (0.1 + getIndex(randoms, 9 + i) * 0.15)}
              />
            ))}
          </svg>
        );

      case "triangles":
        return (
          <svg
            className="absolute inset-0"
            height={size}
            width={size}
            viewBox={`0 0 ${size} ${size}`}
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient
                id={gradientId}
                x1="0%"
                x2="100%"
                y1="0%"
                y2="100%"
              >
                <stop offset="0%" stopColor={color1} />
                <stop offset="50%" stopColor={color2} />
                <stop offset="100%" stopColor={color3} />
              </linearGradient>
            </defs>
            <rect fill={`url(#${gradientId})`} height="100%" width="100%" />
            {Array.from({ length: 8 }).map((_, i) => {
              const x1 = size * getIndex(randoms, 8 + i);
              const y1 = size * getIndex(randoms, 9 + i);
              const x2 = x1 + size * (0.1 + getIndex(randoms, 10 + i) * 0.3);
              const y2 = y1 + size * (0.1 + getIndex(randoms, 11 + i) * 0.3);
              const x3 = x1 + size * (getIndex(randoms, 12 + i) * 0.4 - 0.2);
              const y3 = y1 + size * (0.2 + getIndex(randoms, 13 + i) * 0.3);

              return (
                <polygon
                  fill={i % 3 === 0 ? color1 : i % 3 === 1 ? color2 : color3}
                  key={i}
                  opacity={0.2 + getIndex(randoms, 14 + i) * 0.3}
                  points={`${x1},${y1} ${x2},${y2} ${x3},${y3}`}
                />
              );
            })}
          </svg>
        );

      case "hexagons":
        return (
          <svg
            className="absolute inset-0"
            height={size}
            width={size}
            viewBox={`0 0 ${size} ${size}`}
            preserveAspectRatio="none"
          >
            <defs>
              <radialGradient cx="30%" cy="30%" id={gradientId} r="70%">
                <stop offset="0%" stopColor={color2} />
                <stop offset="60%" stopColor={color1} />
                <stop offset="100%" stopColor={color3} />
              </radialGradient>
            </defs>
            <rect fill={`url(#${gradientId})`} height="100%" width="100%" />
            {Array.from({ length: 12 }).map((_, i) => {
              const centerX = size * (0.1 + getIndex(randoms, 8 + i) * 0.8);
              const centerY = size * (0.1 + getIndex(randoms, 9 + i) * 0.8);
              const radius = size * (0.05 + getIndex(randoms, 10 + i) * 0.1);

              let hexPath = "";
              for (let j = 0; j < 6; j++) {
                const angle = (j * Math.PI) / 3;
                const x = centerX + radius * Math.cos(angle);
                const y = centerY + radius * Math.sin(angle);
                hexPath += j === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
              }
              hexPath += " Z";

              return (
                <path
                  d={hexPath}
                  fill={i % 3 === 0 ? color1 : i % 3 === 1 ? color2 : color3}
                  key={i}
                  opacity={0.3 + getIndex(randoms, 11 + i) * 0.4}
                />
              );
            })}
          </svg>
        );

      default:
        return assertUnreachable(patternType);
    }
  };

  return (
    <div
      className={`
        relative flex items-center justify-center overflow-hidden rounded-full

        ${className}
      `}
      style={{ height: size, width: size }}
    >
      {generatePattern()}
      {showInitials && (
        <span
          className="relative z-10 font-bold text-white drop-shadow-lg"
          style={{
            fontSize: size * 0.35,
            textShadow: "0 1px 3px rgba(0,0,0,0.5)",
          }}
        >
          {initials}
        </span>
      )}
    </div>
  );
}
