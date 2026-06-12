import type {
  CodingToolAvailability,
  DesktopAppMetadata,
  DesktopCommand,
  DesktopNativeStatus,
  DesktopRowContextMenuInput,
  HaloDesktopRPCSchema,
} from "../../desktop/commands";

export const DESKTOP_COMMAND_EVENT = "halo:desktop-command";
export const DESKTOP_NATIVE_STATUS_EVENT = "halo:desktop-native-status";
export const TRACE_PAGE_COMMAND_EVENT = "halo:trace-page-command";

/**
 * True inside the ElectroBun shell (its preload sets __electrobun before app
 * code runs). Used for chrome that only makes sense with a native window,
 * like reserving the macOS traffic-light zone.
 */
export function isDesktopShell(): boolean {
  return (
    typeof window !== "undefined" &&
    Boolean((window as Window & { __electrobun?: unknown }).__electrobun)
  );
}

type DesktopRpc = {
  request: {
    checkForUpdates: () => Promise<DesktopNativeStatus>;
    detectCodingTools: () => Promise<CodingToolAvailability>;
    getAppMetadata: () => Promise<DesktopAppMetadata>;
    openAppDataFolder: () => Promise<{ ok: boolean }>;
    openExternal: (params: { url: string }) => Promise<{ ok: boolean }>;
    revealDatabaseFile: () => Promise<{ ok: boolean }>;
    showNotification: (params: {
      body?: string;
      title: string;
    }) => Promise<{ ok: boolean }>;
    showRowContextMenu: (
      params: DesktopRowContextMenuInput,
    ) => Promise<{ ok: boolean }>;
  };
};

let rpcPromise: Promise<DesktopRpc | null> | undefined;

declare global {
  interface WindowEventMap {
    [DESKTOP_COMMAND_EVENT]: CustomEvent<DesktopCommand>;
    [DESKTOP_NATIVE_STATUS_EVENT]: CustomEvent<DesktopNativeStatus>;
    [TRACE_PAGE_COMMAND_EVENT]: CustomEvent<TracePageCommand>;
  }
}

export type TracePageCommand =
  | { type: "copy-ingest-url" }
  | { type: "open-clear-data" }
  | { type: "open-import" }
  | { type: "refresh" }
  | { type: "toggle-follow-latest" };

export function initializeDesktopBridge() {
  if (rpcPromise) return rpcPromise;

  rpcPromise = (async () => {
    if (typeof window === "undefined") return null;
    const maybeElectrobunWindow = window as Window & {
      __electrobun?: unknown;
      __electrobunWebviewId?: unknown;
    };
    if (!maybeElectrobunWindow.__electrobun) return null;

    const { Electroview } = await import("electrobun/view");
    const rpc = Electroview.defineRPC<HaloDesktopRPCSchema>({
      maxRequestTime: 60_000,
      handlers: {
        requests: {},
        messages: {
          desktopCommand(command) {
            window.dispatchEvent(
              new CustomEvent(DESKTOP_COMMAND_EVENT, { detail: command }),
            );
          },
          nativeStatus(status) {
            window.dispatchEvent(
              new CustomEvent(DESKTOP_NATIVE_STATUS_EVENT, { detail: status }),
            );
          },
        },
      },
    });

    new Electroview({ rpc });
    return rpc as DesktopRpc;
  })();

  return rpcPromise;
}

export async function getDesktopRpc() {
  return initializeDesktopBridge();
}

export function dispatchTracePageCommand(command: TracePageCommand) {
  window.dispatchEvent(
    new CustomEvent(TRACE_PAGE_COMMAND_EVENT, { detail: command }),
  );
}

export async function showDesktopRowContextMenu(
  input: DesktopRowContextMenuInput,
) {
  const rpc = await getDesktopRpc();
  if (!rpc) return false;
  try {
    return (await rpc.request.showRowContextMenu(input)).ok;
  } catch {
    return false;
  }
}

/** Null outside the desktop shell (plain browser dev). */
export async function detectInstalledCodingTools(): Promise<CodingToolAvailability | null> {
  const rpc = await getDesktopRpc();
  if (!rpc) return null;
  try {
    return await rpc.request.detectCodingTools();
  } catch {
    return null;
  }
}

export async function openExternalUrl(url: string) {
  const rpc = await getDesktopRpc();
  if (!rpc) {
    // Browser dev fallback — the browser will ask before opening the scheme.
    window.open(url, "_blank");
    return true;
  }
  try {
    return (await rpc.request.openExternal({ url })).ok;
  } catch {
    return false;
  }
}
