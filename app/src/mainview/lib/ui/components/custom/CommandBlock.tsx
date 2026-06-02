import React from "react";
import { CopyIcon } from "lucide-react";
import { useCopyToClipboard } from "usehooks-ts";

import { Col } from "~/lib/ui/components/custom/Col";
import { Row } from "~/lib/ui/components/custom/Row";
import { Button } from "~/lib/ui/components/ui/Button";
import { Tooltip } from "~/lib/ui/components/ui/Tooltip";
import { toast } from "~/lib/ui/hooks/useToast.hook";

type CommandBlockProps = React.HTMLAttributes<HTMLDivElement> & {
  cmd: string;
  toastDescription?: string;
};

export function CommandBlock({
  cmd,
  toastDescription,
  ...rest
}: CommandBlockProps) {
  const [_, copy] = useCopyToClipboard();

  const handleCopy = () => {
    void copy(cmd);
    toast.success({
      description:
        toastDescription ?? "The command has been copied to your clipboard.",
      title: "Code Copied",
    });
  };

  return (
    <Row
      className={`
        w-full justify-between gap-1 rounded-xl border border-subtle bg-muted
        p-3 text-sm
      `}
      {...rest}
    >
      <Row className="w-full items-center justify-between">
        <pre className="whitespace-break-spaces font-mono">
          <code className="text-detail-success">{cmd}</code>
        </pre>
      </Row>
      <Col className="h-full items-start">
        <Tooltip content="Copy to clipboard">
          <Button
            className="shrink-0"
            onClick={handleCopy}
            size="icon"
            variant="secondary"
          >
            <CopyIcon className="size-4" />
          </Button>
        </Tooltip>
      </Col>
    </Row>
  );
}
