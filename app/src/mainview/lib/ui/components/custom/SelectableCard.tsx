import type { ReactNode } from "react";

import { Label } from "~/lib/ui/components/ui/Label";

type SelectableCardProps = {
  selected: boolean;
  onClick: () => void;
  disabled?: boolean;
  recommended?: boolean;
} & (
  | { title: string; description: string; children?: never }
  | { title?: never; description?: never; children: ReactNode }
);

export function SelectableCard({
  selected,
  onClick,
  disabled = false,
  recommended = false,
  title,
  description,
  children,
}: SelectableCardProps) {
  return (
    <div
      className={`
        rounded-xl border p-4 transition-colors

        ${
          selected
            ? "border-detail-brand"
            : `
              border-card-border

              hover:border-card-border-hover
            `
        }
        ${disabled ? "cursor-not-allowed opacity-70" : "cursor-pointer"}
      `}
      onClick={disabled ? undefined : onClick}
    >
      <div className="flex items-start gap-3">
        <div className="mt-1 flex items-center justify-center">
          <div
            className={`
              h-4 w-4 shrink-0 rounded-full border

              ${
                selected
                  ? "border-detail-brand bg-detail-brand"
                  : "border-card-border"
              }
            `}
          >
            {selected && (
              <div className="h-full w-full scale-50 rounded-full bg-secondary" />
            )}
          </div>
        </div>
        {children ? (
          <div className="min-w-0 flex-1">{children}</div>
        ) : (
          <div className="flex-1">
            <div className="mb-1 flex items-center gap-2">
              <Label className="text-base font-semibold">{title}</Label>
              {recommended && (
                <span
                  className={`
                    rounded border border-detail-brand px-2 py-1 text-xs
                    font-medium text-detail-brand
                  `}
                >
                  RECOMMENDED
                </span>
              )}
            </div>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        )}
      </div>
    </div>
  );
}
