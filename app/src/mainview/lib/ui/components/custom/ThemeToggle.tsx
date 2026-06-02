import React from "react";

import { Button } from "~/lib/ui/components/ui/Button";
import { useTheme } from "~/lib/ui/providers/ThemeProvider";
import { getThemeLabel, THEME_OPTIONS } from "~/lib/ui/theme/themeRegistry";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../ui/DropdownMenu";

type ThemeToggleProps = {
  trigger?: React.ReactNode;
};

export function ThemeToggle({ trigger }: ThemeToggleProps) {
  const { setTheme, theme } = useTheme();

  const themeLabel = getThemeLabel(theme);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        {trigger ?? <Button variant="outline">{themeLabel}</Button>}
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {THEME_OPTIONS.map((option) => {
          return (
            <DropdownMenuItem
              key={option.value}
              onClick={() => setTheme(option.value)}
            >
              {option.label}
            </DropdownMenuItem>
          );
        })}
        <p
          className={`
            mb-1 ml-2 mt-2 hidden text-[10px] text-muted-foreground

            sm:block
          `}
        >
          Or press 't' to toggle
        </p>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
