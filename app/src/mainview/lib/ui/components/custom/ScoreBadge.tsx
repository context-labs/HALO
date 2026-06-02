import { CheckCircleIcon, SquareActivityIcon } from "lucide-react";

import { cn } from "../../utils/utils";
import { Badge } from "../ui/Badge";

export type ScoreBadgeProps = {
  scorePercent: number | null | undefined;
  label?: string;
  showIcon?: boolean;
  className?: string;
};

export function ScoreBadge({
  scorePercent,
  label,
  showIcon = true,
  className,
}: ScoreBadgeProps) {
  if (scorePercent == null) {
    return (
      <Badge
        variant="outline"
        className={cn(
          "border-subtle bg-secondary text-muted-foreground",
          className,
        )}
      >
        {showIcon && <SquareActivityIcon className="mr-1 h-4 w-4" />}
        N/A
      </Badge>
    );
  }

  const isPerfectScore = Math.floor(scorePercent) === 100;
  const isHighScore = scorePercent >= 95;
  const isMediumScore = scorePercent >= 50 && scorePercent < 80;
  const isLowScore = scorePercent < 50;
  const formattedScore = scorePercent.toFixed(2);

  return (
    <Badge
      variant="outline"
      className={cn(
        "border-subtle bg-secondary",
        isHighScore &&
          "border-detail-success bg-detail-success/10 text-detail-success",
        isMediumScore &&
          "border-detail-warning bg-detail-warning/10 text-detail-warning",
        isLowScore &&
          "border-detail-failure bg-detail-failure/10 text-detail-failure",
        className,
      )}
    >
      {showIcon &&
        (isPerfectScore ? (
          <CheckCircleIcon className="mr-1 h-4 w-4" />
        ) : (
          <SquareActivityIcon className="mr-1 h-4 w-4" />
        ))}
      {label
        ? `${label} ${isPerfectScore ? "100%" : `${formattedScore}%`}`
        : isPerfectScore
          ? "100%"
          : `${formattedScore}%`}
    </Badge>
  );
}
