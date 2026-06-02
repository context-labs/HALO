import { useTheme } from "~/lib/ui/providers/ThemeProvider";

export function HomePageBackdrop() {
  const { resolvedTheme: theme } = useTheme();

  if (theme !== "dark") {
    return null;
  }

  return (
    <div
      className="fixed inset-0 -z-10"
      style={{
        background: `
            radial-gradient(ellipse 80% 80% at 50% -20%, rgba(120, 200, 255, 0.15), transparent),
            radial-gradient(ellipse 70% 40% at 80% 50%, rgba(120, 255, 180, 0.08), transparent),
            radial-gradient(ellipse 90% 60% at 20% 100%, rgba(100, 180, 255, 0.06), transparent),
            linear-gradient(180deg,
              #0a0a0a 0%,
              #0f0f11 20%,
              #131316 50%,
              #0a0a0c 100%)
          `,
      }}
    />
  );
}
