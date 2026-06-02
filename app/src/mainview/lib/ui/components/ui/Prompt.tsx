import type { ButtonVariantProp } from "~/lib/ui";
import {
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogRoot,
  AlertDialogTitle,
  AlertDialogTrigger,
  Button,
} from "~/lib/ui";

type PromptProps = {
  confirmText?: string;
  description?: React.ReactNode;
  key?: string;
  loading?: boolean;
  onCancel?: () => void;
  onConfirm: () => void;
  onOpenChange?: (open: boolean) => void;
  open?: boolean;
  title?: string;
  confirmButtonVariant?: ButtonVariantProp;
  trigger: React.ReactElement | string;
};

export function Prompt(props: PromptProps) {
  const {
    confirmButtonVariant,
    confirmText = "Confirm",
    description,
    key,
    loading = false,
    onCancel = () => undefined,
    onConfirm,
    onOpenChange,
    open,
    title = "Are you sure?",
    trigger,
  } = props;
  const isControlledComponent = open != null && onOpenChange != null;
  return (
    <AlertDialogRoot key={key} onOpenChange={onOpenChange} open={open}>
      <AlertDialogTrigger asChild>
        {typeof trigger === "string" ? (
          <Button variant="outline">{trigger}</Button>
        ) : (
          trigger
        )}
      </AlertDialogTrigger>
      <AlertDialogContent>
        <form
          onSubmit={(e) => {
            onConfirm();
            e.preventDefault();
          }}
        >
          <AlertDialogHeader>
            <AlertDialogTitle>{title}</AlertDialogTitle>
            {description && (
              <AlertDialogDescription className="py-2">
                {description}
              </AlertDialogDescription>
            )}
            {description == null && <div className="h-8" />}
          </AlertDialogHeader>
          <AlertDialogFooter className="mt-2 gap-2">
            <AlertDialogCancel disabled={loading} onClick={onCancel}>
              Cancel
            </AlertDialogCancel>
            {isControlledComponent ? (
              <Button disabled={loading} variant={confirmButtonVariant}>
                {loading ? "Loading..." : confirmText}
              </Button>
            ) : (
              <AlertDialogAction
                disabled={loading}
                type="submit"
                variant={confirmButtonVariant}
              >
                {loading ? "Loading..." : confirmText}
              </AlertDialogAction>
            )}
          </AlertDialogFooter>
        </form>
      </AlertDialogContent>
    </AlertDialogRoot>
  );
}
