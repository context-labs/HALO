import { useEffect } from "react";
import { AlertTriangleIcon, CheckCircle, InfoIcon } from "lucide-react";

import { Row } from "~/lib/ui/components/custom/Row";
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "~/lib/ui/components/ui/Toast";
import { useToast } from "~/lib/ui/hooks/useToast.hook";

const TOAST_REMOVE_ON_WINDOW_BLUR_DELAY_MS = 500;

export function Toaster() {
  const { dismiss, toasts } = useToast();

  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = [];
    const handleBlur = () => {
      toasts.forEach((toast) => {
        const timer = setTimeout(() => {
          dismiss(toast.id);
        }, TOAST_REMOVE_ON_WINDOW_BLUR_DELAY_MS);
        timers.push(timer);
      });
    };

    window.addEventListener("blur", handleBlur);
    return () => {
      window.removeEventListener("blur", handleBlur);
      timers.forEach((timer) => clearTimeout(timer));
    };
  }, [dismiss, toasts]);

  return (
    <ToastProvider>
      {toasts.map(function ({
        action,
        description,
        id,
        title,
        variant,
        ...props
      }) {
        return (
          <Toast key={id} variant={variant} {...props}>
            <div className="grid gap-1">
              {title && (
                <Row className="gap-[6px]">
                  {variant === "info" && (
                    <InfoIcon size={18} className="shrink-0" />
                  )}
                  {variant === "success" && (
                    <CheckCircle
                      size={18}
                      className="shrink-0 text-detail-success"
                    />
                  )}
                  {variant === "warning" && (
                    <AlertTriangleIcon
                      size={18}
                      className="shrink-0 text-detail-warning"
                    />
                  )}
                  {variant === "destructive" && (
                    <AlertTriangleIcon
                      size={18}
                      className="shrink-0 text-detail-failure"
                    />
                  )}
                  <ToastTitle>{title}</ToastTitle>
                </Row>
              )}
              {description && (
                <ToastDescription>{description}</ToastDescription>
              )}
            </div>
            {action}
            <ToastClose />
          </Toast>
        );
      })}
      <ToastViewport />
    </ToastProvider>
  );
}
