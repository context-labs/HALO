export const THEME_OPTIONS = [
  { value: "system", label: "System", mode: "system" },
  { value: "light", label: "Light", mode: "light" },
  { value: "dark", label: "Dark", mode: "dark" },
] as const;

export type Theme = (typeof THEME_OPTIONS)[number]["value"];
export type CoreTheme = "light" | "dark";
export type ResolvedTheme = Exclude<Theme, "system">;
export type ThemeMode = (typeof THEME_OPTIONS)[number]["mode"];

const THEME_SET = new Set<string>(THEME_OPTIONS.map((theme) => theme.value));

export function isTheme(value: string): value is Theme {
  return THEME_SET.has(value);
}

export function resolveTheme(
  theme: Theme,
  systemTheme: CoreTheme,
): ResolvedTheme {
  return theme === "system" ? systemTheme : theme;
}

export function toCoreTheme(theme: ResolvedTheme): CoreTheme {
  return theme === "dark" ? "dark" : "light";
}

export function getThemeClasses(theme: ResolvedTheme): string[] {
  return [theme];
}

export function getThemeLabel(theme: Theme): string {
  const match = THEME_OPTIONS.find((option) => option.value === theme);
  return match?.label ?? "System";
}
