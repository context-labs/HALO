import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

import type { CoreTheme, ResolvedTheme, Theme } from "../theme/themeRegistry";
import {
  getThemeClasses,
  isTheme,
  resolveTheme,
  toCoreTheme,
} from "../theme/themeRegistry";

export type { Theme };

const DEFAULT_THEME: Theme = "system";

type ThemeProviderProps = {
  children: React.ReactNode;
  storageKey: string;
  enableMarketingBehavior?: boolean;
  storage: Storage;
};

type ThemeProviderState = {
  theme: Theme;
  isDarkTheme: boolean;
  isLightTheme: boolean;
  appliedTheme?: Theme;
  setTheme: (theme: Theme) => void;
  resolvedTheme: ResolvedTheme;
  darkOrLightTheme: CoreTheme;
};

const initialState: ThemeProviderState = {
  resolvedTheme: "dark" as const,
  setTheme: () => null,
  theme: "system" as const,
  darkOrLightTheme: "dark" as const,
  isDarkTheme: true,
  isLightTheme: false,
};

const ThemeProviderContext = createContext<ThemeProviderState>(initialState);

const getSystemTheme = (): CoreTheme => {
  if (typeof window === "undefined") {
    return "dark";
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
};

export function ThemeProvider({
  children,
  enableMarketingBehavior = false,
  storage,
  storageKey,
  ...props
}: ThemeProviderProps) {
  const [appliedTheme, setAppliedTheme] = useState<Theme | undefined>();
  const [theme, setTheme] = useState<Theme>(
    () => (storage.getItem(storageKey) ?? DEFAULT_THEME) as Theme,
  );
  const [systemTheme, setSystemTheme] = useState<CoreTheme>(getSystemTheme);

  const updateDocumentBodyTheme = useCallback(
    (themeToApply: Theme) => {
      const root = window.document.body;
      root.classList.remove("light", "dark");
      const resolvedTheme = resolveTheme(themeToApply, systemTheme);
      root.classList.add(...getThemeClasses(resolvedTheme));
      root.classList.add("theme-ready");
    },
    [systemTheme],
  );

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const handleSystemThemeChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? "dark" : "light");
    };

    mediaQuery.addEventListener("change", handleSystemThemeChange);

    return () => {
      mediaQuery.removeEventListener("change", handleSystemThemeChange);
    };
  }, []);

  useEffect(() => {
    const resolvedTheme = resolveTheme(theme, systemTheme);
    if (enableMarketingBehavior && resolvedTheme === "light") {
      setAppliedTheme("dark");
      updateDocumentBodyTheme("dark");
    } else if (appliedTheme !== resolvedTheme) {
      setAppliedTheme(undefined);
      updateDocumentBodyTheme(theme);
    }
  }, [
    appliedTheme,
    enableMarketingBehavior,
    theme,
    systemTheme,
    updateDocumentBodyTheme,
  ]);

  useEffect(() => {
    updateDocumentBodyTheme(theme);
  }, [theme, systemTheme, updateDocumentBodyTheme]);

  const resolvedTheme = resolveTheme(appliedTheme ?? theme, systemTheme);

  const darkOrLightTheme: CoreTheme = toCoreTheme(resolvedTheme);

  const state: ThemeProviderState = {
    appliedTheme,
    resolvedTheme,
    setTheme: (theme: Theme) => {
      storage.setItem(storageKey, theme);
      setTheme(theme);
    },
    theme,
    darkOrLightTheme,
    isDarkTheme: darkOrLightTheme === "dark",
    isLightTheme: darkOrLightTheme === "light",
  };

  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === storageKey && event.newValue) {
        if (!isTheme(event.newValue)) {
          return;
        }
        setTheme(event.newValue);
      }
    };

    window.addEventListener("storage", handleStorageChange);
    return () => {
      window.removeEventListener("storage", handleStorageChange);
    };
  }, [storageKey]);

  return (
    <ThemeProviderContext.Provider {...props} value={state}>
      {children}
    </ThemeProviderContext.Provider>
  );
}

export function useTheme(): ThemeProviderState {
  const context = useContext(ThemeProviderContext);

  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  if (context == undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }

  const resolvedTheme = context.appliedTheme
    ? resolveTheme(context.appliedTheme, getSystemTheme())
    : context.resolvedTheme;

  const darkOrLightTheme: CoreTheme = toCoreTheme(resolvedTheme);

  return {
    ...context,
    isDarkTheme: darkOrLightTheme === "dark",
    isLightTheme: darkOrLightTheme === "light",
    resolvedTheme,
    theme: context.theme,
    darkOrLightTheme,
  };
}
