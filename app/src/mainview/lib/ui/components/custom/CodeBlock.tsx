import type { ClassNameValue } from "tailwind-merge";
import { useCallback, useState } from "react";
import { Copy } from "lucide-react";
import SyntaxHighlighter from "react-syntax-highlighter";
import a11yLight from "react-syntax-highlighter/dist/esm/styles/hljs/a11y-light.js";
import atomOneDark from "react-syntax-highlighter/dist/esm/styles/hljs/atom-one-dark.js";
import { useCopyToClipboard } from "usehooks-ts";

import { Row } from "~/lib/ui/components/custom/Row";
import { Button } from "~/lib/ui/components/ui/Button";
import { Card } from "~/lib/ui/components/ui/Card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "~/lib/ui/components/ui/Tabs";
import { Tooltip } from "~/lib/ui/components/ui/Tooltip";
import { toast } from "~/lib/ui/hooks/useToast.hook";
import { useTheme } from "~/lib/ui/providers/ThemeProvider";
import { cn } from "~/lib/ui/utils/utils";

type CopyCodeButtonProps = {
  onCopy: () => void;
};

function CopyCodeButton({ onCopy }: CopyCodeButtonProps) {
  return (
    <Tooltip content="Copy to clipboard">
      <Button
        aria-label="Copy code to clipboard"
        className="group bg-secondary/80"
        onClick={onCopy}
        size="icon"
        variant="ghost"
      >
        <Copy
          className={`
            size-4 text-muted-foreground

            group-hover:text-foreground
          `}
        />
      </Button>
    </Tooltip>
  );
}

export type CodeBlockTab = {
  label: string;
  code: string;
  obfuscatedCode: string;
  language?: string;
};

type SharedProps = {
  className?: ClassNameValue;
  cta?: React.ReactNode;
  copyButton?: React.ReactNode;
  customStyle?: React.CSSProperties;
  onCopy?: () => void;
  singleLine?: boolean;
};

type CodeBlockTabsProps = SharedProps & {
  language?: string;
  tabs: CodeBlockTab[];
};

type CodeBlockCodeProps = SharedProps & {
  code: string;
  obfuscatedCode: string;
  language: string;
};

type CodeBlockProps = CodeBlockTabsProps | CodeBlockCodeProps;

type CodeContentProps = {
  copyButton?: React.ReactNode;
  code: string;
  obfuscatedCode: string;
  language: string;
  handleCopy: (code: string) => void;
  customStyle?: React.CSSProperties;
  singleLine?: boolean;
};

export function CodeContent({
  code,
  copyButton,
  customStyle,
  handleCopy,
  language = "plaintext",
  obfuscatedCode,
  singleLine,
}: CodeContentProps) {
  const { isDarkTheme } = useTheme();
  return (
    <div
      className={`
        relative h-full w-auto rounded-md p-2

      `}
    >
      {copyButton && (
        <div
          className={cn("absolute right-2 ", singleLine ? "top-1" : "top-2")}
        >
          <CopyCodeButton onCopy={() => handleCopy(code)} />
        </div>
      )}
      <SyntaxHighlighter
        codeTagProps={{ style: { fontFamily: "inherit" } }}
        customStyle={customStyle}
        id="CodeContent"
        language={language}
        style={{
          ...(isDarkTheme ? atomOneDark : a11yLight),
          hljs: {
            ...(isDarkTheme ? atomOneDark.hljs : a11yLight.hljs),
            background: "transparent",
          },
        }}
        wrapLongLines={false}
      >
        {obfuscatedCode}
      </SyntaxHighlighter>
    </div>
  );
}

export function CodeBlock({
  className,
  copyButton = <></>,
  cta,
  customStyle,
  language,
  onCopy,
  singleLine,
  ...rest
}: CodeBlockProps) {
  const [, copyToClipboard] = useCopyToClipboard();
  const [tab, setTab] = useState<string>(
    ("tabs" in rest ? rest.tabs[0]?.label : "") ?? "",
  );

  const handleCopy = useCallback(
    (codeToClip: string) => {
      if (codeToClip) {
        void copyToClipboard(codeToClip);
        toast.success({
          description: "Code snippet has been copied to your clipboard.",
          title: "Code Copied",
        });
        onCopy?.();
      }
    },
    [copyToClipboard, onCopy],
  );

  const code = "code" in rest ? rest.code : null;
  if (code) {
    const obfuscatedCode = "obfuscatedCode" in rest ? rest.obfuscatedCode : "";
    return (
      <Card className={cn("relative flex flex-col", className)}>
        {code && (
          <CodeContent
            code={code}
            copyButton={copyButton}
            customStyle={customStyle}
            handleCopy={handleCopy}
            language={language ?? "plaintext"}
            obfuscatedCode={obfuscatedCode}
            singleLine={singleLine}
          />
        )}
      </Card>
    );
  }

  const tabs = "tabs" in rest ? rest.tabs : [];
  return (
    <Card
      className={cn(
        `
          relative flex flex-col

        `,
        className,
      )}
    >
      <Tabs onValueChange={setTab} value={tab}>
        <TabsList className="w-full rounded-none border-b bg-card px-2 py-[6px]">
          <Row className="w-full justify-between">
            <Row className="gap-2">
              {tabs.map((tab) => (
                <TabsTrigger key={tab.label} value={tab.label}>
                  {tab.label}
                </TabsTrigger>
              ))}
            </Row>
            {cta != null && cta}
          </Row>
        </TabsList>
        {tabs.map((tab) => (
          <TabsContent key={tab.label} value={tab.label}>
            <CodeContent
              code={tab.code}
              copyButton={copyButton}
              customStyle={customStyle}
              handleCopy={handleCopy}
              language={tab.language ?? "plaintext"}
              obfuscatedCode={tab.obfuscatedCode}
              singleLine={singleLine}
            />
          </TabsContent>
        ))}
      </Tabs>
    </Card>
  );
}
