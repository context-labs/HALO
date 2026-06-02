import * as React from "react";

import { Button } from "~/lib/ui/components/ui/Button";
import { cn } from "~/lib/ui/utils/utils";

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    className={cn(
      "rounded-xl border border-subtle bg-card text-card-foreground",
      className,
    )}
    ref={ref}
    {...props}
  />
));
Card.displayName = "Card";

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    ref={ref}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h4 className={cn(className)} ref={ref} {...props} />
));
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    className={cn("text-sm text-muted-foreground", className)}
    ref={ref}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div className={cn("p-6 pt-0", className)} ref={ref} {...props} />
));
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    className={cn("flex items-center p-6 pt-0", className)}
    ref={ref}
    {...props}
  />
));
CardFooter.displayName = "CardFooter";

type CardProps = {
  title?: React.ReactNode;
  description?: React.ReactNode;
  children?: React.ReactNode;
  primaryButton?: {
    label: string;
    onClick: () => void;
  };
  secondaryButton?: {
    label: string;
    onClick: () => void;
  };
};

type CombinedCardProps = CardProps &
  Omit<React.ComponentPropsWithoutRef<typeof Card>, keyof CardProps>;

const CardComponent = React.forwardRef<HTMLDivElement, CombinedCardProps>(
  (
    {
      children,
      className,
      description,
      primaryButton,
      secondaryButton,
      title,
      ...props
    },
    ref,
  ) => {
    return (
      <Card className={cn(className)} ref={ref} {...props}>
        {(title ?? description) && (
          <CardHeader>
            {title && <CardTitle>{title}</CardTitle>}
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
        )}
        {children && <CardContent>{children}</CardContent>}
        {(primaryButton ?? secondaryButton) && (
          <CardFooter className="flex justify-between">
            {secondaryButton && (
              <Button onClick={secondaryButton.onClick} variant={"outline"}>
                {secondaryButton.label}
              </Button>
            )}
            {primaryButton && (
              <Button onClick={primaryButton.onClick} variant={"default"}>
                {primaryButton.label}
              </Button>
            )}
          </CardFooter>
        )}
      </Card>
    );
  },
);
CardComponent.displayName = "Card";

export {
  Card,
  CardComponent,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
};
