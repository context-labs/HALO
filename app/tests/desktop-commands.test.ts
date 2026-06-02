import { describe, expect, test } from "bun:test";
import {
  commandLabel,
  desktopCommandForShortcut,
  filterCommandPaletteItems,
  routeForCommand,
} from "../src/desktop/commands";

describe("desktop command helpers", () => {
  test("maps workspace menu commands to routes", () => {
    expect(routeForCommand("navigate-traces")).toBe("traces");
    expect(routeForCommand("navigate-sessions")).toBe("sessions");
    expect(routeForCommand("navigate-analysis")).toBe("analysis");
    expect(routeForCommand("navigate-settings")).toBe("settings");
    expect(routeForCommand("preferences")).toBe("settings");
    expect(routeForCommand("copy-ingest-url")).toBeUndefined();
  });

  test("maps keyboard shortcuts to app commands", () => {
    expect(desktopCommandForShortcut("k")).toBe("command-palette");
    expect(desktopCommandForShortcut(",")).toBe("preferences");
    expect(desktopCommandForShortcut("1")).toBe("navigate-traces");
    expect(desktopCommandForShortcut("2")).toBe("navigate-sessions");
    expect(desktopCommandForShortcut("3")).toBe("navigate-analysis");
    expect(desktopCommandForShortcut("4")).toBe("navigate-settings");
    expect(desktopCommandForShortcut("r")).toBe("refresh");
    expect(desktopCommandForShortcut("c", true)).toBe("copy-ingest-url");
    expect(desktopCommandForShortcut("i", true)).toBe("import-data");
    expect(desktopCommandForShortcut("l", true)).toBe("toggle-follow-latest");
    expect(desktopCommandForShortcut("c")).toBeUndefined();
  });

  test("filters command palette items by label and keywords", () => {
    expect(filterCommandPaletteItems("langfuse").map((item) => item.command)).toEqual([
      "import-data",
    ]);
    expect(filterCommandPaletteItems("provider").map((item) => item.command)).toContain(
      "preferences",
    );
    expect(commandLabel("copy-ingest-url")).toBe("Copy Ingest URL");
  });
});
