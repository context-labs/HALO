import { InfoIcon } from "lucide-react";

import { Alert } from "~/lib/ui/components/ui/Alert";

type AlertInfoProps = {
  title: string;
  content?: string | React.ReactNode;
  className?: string;
};

export function AlertInfo({ className, content, title }: AlertInfoProps) {
  return (
    <Alert className={className} variant="info">
      <div className="flex items-start gap-3">
        <div className="mt-0.5">
          <InfoIcon className="h-5 w-5 text-detail-brand" />
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
