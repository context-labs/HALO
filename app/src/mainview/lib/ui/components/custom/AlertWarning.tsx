import { AlertTriangleIcon } from "lucide-react";

import { Alert } from "~/lib/ui/components/ui/Alert";

type AlertWarningProps = {
  title: string;
  content?: string | React.ReactNode;
  className?: string;
};

export function AlertWarning({ className, content, title }: AlertWarningProps) {
  return (
    <Alert className={className} variant="warning">
      <div className="flex items-start gap-3">
        <div className="mt-0.5">
          <AlertTriangleIcon className="h-5 w-5 text-detail-warning" />
        </div>
        <div className="flex-1">
          <h6 className="font-semibold text-foreground">{title}</h6>
          {content && (
            <div className="mt-1">
              {typeof content === "string" ? (
                <p className="text-sm text-muted-foreground">{content}</p>
              ) : (
                content
              )}
            </div>
          )}
        </div>
      </div>
    </Alert>
  );
}
