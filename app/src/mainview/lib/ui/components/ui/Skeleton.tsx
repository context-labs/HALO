import { cn } from "~/lib/ui/utils/utils";

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden="true"
      className={cn("animate-pulse rounded-xl bg-skeleton", className)}
      {...props}
    />
  );
}

function SkeletonText({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <Skeleton className={cn("h-4", className)} {...props} />;
}

function SkeletonButton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <Skeleton className={cn("h-10", className)} {...props} />;
}

function SkeletonAvatar({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <Skeleton className={cn("h-8 w-8", className)} {...props} />;
}

function SkeletonCard({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-subtle bg-card p-6",
        className,
      )}
      {...props}
    >
      <div className="flex items-start gap-4">
        <Skeleton className="h-10 w-10" />
        <div className="flex-1 space-y-2">
          <SkeletonText className="w-1/3" />
          <SkeletonText className="w-1/2" />
        </div>
      </div>
    </div>
  );
}

export { Skeleton, SkeletonText, SkeletonButton, SkeletonAvatar, SkeletonCard };
