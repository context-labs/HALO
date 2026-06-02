import type { ClassValue } from "clsx";
import * as React from "react";

import { cn } from "~/lib/ui/utils/utils";

const Table = React.forwardRef<
  HTMLTableElement,
  React.HTMLAttributes<HTMLTableElement> & {
    wrapperClassName?: ClassValue;
  }
>(({ className, wrapperClassName, ...props }, ref) => (
  <div
    className={cn(
      "relative w-full overflow-hidden rounded-xl border-subtle",
      wrapperClassName,
    )}
  >
    <div className="overflow-auto">
      <table
        className={cn("w-full caption-bottom text-sm", className)}
        ref={ref}
        {...props}
      />
    </div>
  </div>
));
Table.displayName = "Table";

const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead
    className={cn(
      `
        bg-muted/30

        [&_tr]:border-b [&_tr]:border-subtle
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TableHeader.displayName = "TableHeader";

const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement> & {
    alternatingRows?: boolean;
  }
>(({ className, alternatingRows, ...props }, ref) => (
  <tbody
    className={cn(
      alternatingRows &&
        "[&_tr:nth-child(even)]:bg-muted/50! [&_tr:nth-child(odd)]:bg-transparent!",
      className,
    )}
    ref={ref}
    {...props}
  />
));
TableBody.displayName = "TableBody";

const TableFooter = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tfoot
    className={cn(
      `
        border-t bg-muted/50 font-medium

        last:[&>tr]:border-b-0
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TableFooter.displayName = "TableFooter";

const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    className={cn(
      `
        border-b border-subtle bg-transparent text-muted-foreground transition-colors
        last:border-b-0

        data-[state=selected]:bg-muted

        hover:bg-muted/50
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TableRow.displayName = "TableRow";

const TableHead = React.forwardRef<
  HTMLTableCellElement,
  React.ThHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <th
    className={cn(
      `
        h-12 px-4 text-left align-middle font-medium text-foreground

        [&:has([role=checkbox])]:pr-0
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TableHead.displayName = "TableHead";

const TableCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <td
    className={cn(
      `
        p-4 align-middle

        [&:has([role=checkbox])]:pr-0
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TableCell.displayName = "TableCell";

const TableCaption = React.forwardRef<
  HTMLTableCaptionElement,
  React.HTMLAttributes<HTMLTableCaptionElement>
>(({ className, ...props }, ref) => (
  <caption
    className={cn("mt-4 text-sm text-muted-foreground", className)}
    ref={ref}
    {...props}
  />
));
TableCaption.displayName = "TableCaption";

export {
  Table,
  TableHeader,
  TableBody,
  TableFooter,
  TableHead,
  TableRow,
  TableCell,
  TableCaption,
};
