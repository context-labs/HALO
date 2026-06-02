import type { JsonViewProps } from "@uiw/react-json-view";
import JsonView from "@uiw/react-json-view";
import { lightTheme } from "@uiw/react-json-view/light";
import { vscodeTheme } from "@uiw/react-json-view/vscode";

import { useTheme } from "~/lib/ui/providers/ThemeProvider";

type JsonComponentProps = {
  data: object | null | undefined;
} & Omit<JsonViewProps<object>, "value" | "style">;

export function JsonComponent({ data, ...rest }: JsonComponentProps) {
  const { isDarkTheme } = useTheme();
  if (data == null) {
    return <p>No data present.</p>;
  }
  return (
    <JsonView
      style={isDarkTheme ? vscodeTheme : lightTheme}
      value={data}
      {...rest}
    />
  );
}
