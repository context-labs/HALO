import type { ReactNode } from "react";
import { Activity, ArrowRight, FileUp, Terminal } from "lucide-react";

import {
  Badge,
  Button,
  CommandBlock,
  Dialog,
  DialogClose,
  cn,
} from "~/lib/ui";

export function ImportDataScreen({
  onConnectLocalAgent,
  onImportJsonl,
  onImportLangfuse,
  onImportPhoenix,
}: {
  onConnectLocalAgent: () => void;
  onImportJsonl: () => void;
  onImportLangfuse: () => void;
  onImportPhoenix: () => void;
}) {
  return (
    <div className="flex h-full min-h-[calc(100vh-3.5rem)] items-center justify-center overflow-auto p-8">
      <div className="w-full max-w-3xl">
        <div className="mb-8 text-center">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-detail-brand">
            HALO setup
          </p>
          <h1 className="mt-3 text-3xl font-semibold tracking-normal">
            Bring your agent traces into HALO
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-base text-muted-foreground">
            Import historical traces or connect a live agent, then let HALO's
            analysis find the failures and bottlenecks for you.
          </p>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <ImportDataActionCard
            badge="Recommended"
            description="Bring historical traces from a Langfuse project into this local HALO timeline."
            estimatedTime="Est time: 2-5 minutes"
            highlighted
            icon={
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-background-muted">
                <LangfuseLogo className="h-7 w-7" />
              </span>
            }
            onClick={onImportLangfuse}
            title="Import from Langfuse"
          />
          <ImportDataActionCard
            description="Bring historical traces from an Arize Phoenix project into this local HALO timeline."
            estimatedTime="Est time: 2-5 minutes"
            icon={
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-background-muted">
                <PhoenixLogo className="h-7 w-7" />
              </span>
            }
            onClick={onImportPhoenix}
            title="Import from Phoenix"
          />
          <ImportDataActionCard
            description="Upload a JSONL trace export. One span per line, the format HALO and Catalyst exports use."
            estimatedTime="Est time: 1-2 minutes"
            icon={
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-detail-brand/10 text-detail-brand">
                <FileUp className="h-5 w-5" />
              </span>
            }
            onClick={onImportJsonl}
            title="Import JSONL File"
          />
          <ImportDataActionCard
            description="Point a Catalyst or OpenTelemetry JSON exporter at HALO and watch traces stream live."
            estimatedTime="Est time: 2-5 minutes"
            icon={
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-detail-brand/10 text-detail-brand">
                <Activity className="h-5 w-5" />
              </span>
            }
            onClick={onConnectLocalAgent}
            title="Connect Local Agent"
          />
        </div>
      </div>
    </div>
  );
}

export function LocalAgentSetupDialog({
  envLine,
  ingestUrl,
  onOpenChange,
  open,
}: {
  envLine: string;
  ingestUrl: string;
  onOpenChange: (open: boolean) => void;
  open: boolean;
}) {
  return (
    <Dialog
      className="!w-[min(760px,92vw)] !max-w-[92vw] sm:!max-w-[760px] md:!w-[760px]"
      dialogDescription="Send local OTLP or JSON traces to HALO and watch the monitor populate live."
      dialogTitle={
        <span className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-detail-brand" />
          Connect Local Agent
        </span>
      }
      footer={
        <div className="flex justify-end border-t border-subtle px-6 py-4">
          <DialogClose asChild>
            <Button variant="secondary">Done</Button>
          </DialogClose>
        </div>
      }
      hideConfirmButton
      maxWidth={760}
      onConfirm={() => undefined}
      onOpenChange={onOpenChange}
      open={open}
    >
      <div className="space-y-5">
        <div className="rounded-xl border border-subtle bg-background-muted p-4">
          <div className="flex items-start gap-3">
            <Terminal className="mt-0.5 h-4 w-4 shrink-0 text-detail-brand" />
            <div>
              <p className="font-medium">Send traces to the ingest endpoint</p>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">
                Use this endpoint in a local Catalyst agent, OpenTelemetry
                collector, or JSON trace exporter. The first accepted span
                switches this screen back to the live trace monitor.
              </p>
            </div>
          </div>
        </div>

        <SetupCommandRow
          label="Environment variable"
          toastDescription="Paste this into your local agent environment."
          value={envLine}
        />
        <SetupCommandRow
          label="Ingest endpoint"
          toastDescription="Paste this into your local agent telemetry config."
          value={ingestUrl}
        />

        <div className="rounded-xl border border-subtle bg-background-muted p-4">
          <p className="font-medium">Smoke test</p>
          <CommandBlock
            className="mt-3 bg-background"
            cmd="bun run fire:test-spans"
            toastDescription="Run this from the HALO repo after local services are up."
            wrap={false}
          />
          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            Run the test flow once your local services are up to send a sample
            trace batch into HALO.
          </p>
        </div>
      </div>
    </Dialog>
  );
}

function ImportDataActionCard({
  badge,
  description,
  estimatedTime,
  highlighted,
  icon,
  onClick,
  title,
}: {
  badge?: string;
  description: string;
  estimatedTime: string;
  highlighted?: boolean;
  icon: ReactNode;
  onClick: () => void;
  title: string;
}) {
  return (
    <button
      className={cn(
        "group flex min-h-48 w-full flex-col rounded-[18px] border border-border/70 bg-card p-5 text-left transition hover:border-border hover:bg-card-hover/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        highlighted && "border-detail-brand/30 ring-1 ring-detail-brand/20",
      )}
      onClick={onClick}
      type="button"
    >
      <div className="flex w-full items-start justify-between gap-3">
        {icon}
        {badge ? (
          <Badge size="sm" variant="status-brand">
            {badge}
          </Badge>
        ) : null}
      </div>
      <h2 className="mt-5 flex items-center gap-1.5 text-xl font-semibold tracking-normal">
        {title}
        <ArrowRight className="h-4 w-4 -translate-x-1 text-muted-foreground opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100" />
      </h2>
      <p className="mt-2 max-w-xl text-base leading-7 text-muted-foreground">
        {description}
      </p>
      <p className="mt-auto pt-5 text-sm font-medium text-muted-foreground">
        {estimatedTime}
      </p>
    </button>
  );
}

export function LangfuseLogo({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 512 512"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M254.75 302.25L285.25 326.75C285.25 326.75 308.587 309.418 325.75 306.875C343.75 304.208 362.954 314.244 380.75 326.208C407.629 344.279 430.25 367.208 430.25 367.208L456.75 341.208C456.75 341.208 383.686 262.047 325.75 269.208C287.75 273.905 254.75 302.25 254.75 302.25Z"
        fill="#FF5D5F"
      />
      <path
        d="M80.25 151.286L55.25 178.786C55.25 178.786 124.902 243.786 179.75 243.786C204.75 243.786 239.419 224.201 269.25 198.757C286.25 184.257 305.25 167.786 324.25 167.786C337.021 167.786 353.866 174.551 369.75 192.316C369.75 192.316 380.003 186.168 386.25 181.75C391.74 177.868 399.896 171.25 399.896 171.25C377.047 146.864 343.998 129.038 324.25 130.786C292.25 130.79 269.25 150.711 240.75 173.75C212.25 196.789 200.25 206.286 179.75 206.286C145.25 206.286 80.25 151.286 80.25 151.286Z"
        fill="#4E9CFF"
      />
      <path
        d="M80.25 360.75L55.25 333.25C55.25 333.25 124.902 268.25 179.75 268.25C204.75 268.25 239.419 287.835 269.25 313.279C286.25 327.779 305.25 344.25 324.25 344.25C337.083 344.25 353.799 337.207 369.75 319.25C369.75 319.25 379.339 325.161 385.25 329.25C391.328 333.455 400.25 340.407 400.25 340.407C377.39 364.987 344.1 383.007 324.25 381.25C292.25 381.246 273.25 364.289 244.75 341.25C216.25 318.211 200.25 305.75 179.75 305.75C145.25 305.75 80.25 360.75 80.25 360.75Z"
        fill="#4E9CFF"
      />
      <path
        d="M406.25 213.25C399.745 217.746 389.25 224.25 389.25 224.25C389.25 224.25 395.25 237.25 395.25 254.75C395.25 272.25 389.75 287.25 389.75 287.25C389.75 287.25 399.172 293.135 405.25 297.25C411.564 301.525 421.25 308.75 421.25 308.75C421.25 308.75 432.75 284.75 432.75 254.75C432.75 224.75 421.25 202.25 421.25 202.25C421.25 202.25 412.226 209.12 406.25 213.25Z"
        fill="#4E9CFF"
      />
      <path
        d="M256.25 209.25L285.25 185.25C285.25 185.25 308.587 202.04 325.75 204.583C343.75 207.25 362.954 197.214 380.75 185.25C407.629 167.179 430.25 144.25 430.25 144.25L456.75 170.25C456.75 170.25 383.686 249.411 325.75 242.25C287.75 237.553 256.25 209.25 256.25 209.25Z"
        fill="#FF5D5F"
      />
      <path
        d="M186.255 130.25C223.755 130.25 255.25 162.25 255.25 162.25C255.25 162.25 246.487 169.155 240.75 173.75C234.775 178.536 225.25 186.25 225.25 186.25C225.25 186.25 208.755 168.75 186.255 168.75C177.028 168.75 165.039 174.292 152.255 185.25C142.391 193.705 132.129 204.216 125.255 217.25C119.31 228.52 116.068 241.802 115.755 255.75C115.361 273.269 121.571 291.634 131.755 306.25C138.58 316.046 146.726 323.418 155.255 329.75C166.323 337.968 177.865 343.75 186.255 343.75C195.217 343.75 203.274 340.635 209.255 337.75C218.755 332.25 226.25 325.75 226.25 325.75L255.75 350.25C255.75 350.25 243.75 362.25 227.255 371.25C216.595 376.507 202.895 381.75 186.255 381.75C169.626 381.75 150.315 372.915 132.255 359.25C120.579 350.416 109.135 339.948 100.255 327.25C85.7005 306.438 78.2004 281.118 78.2502 255.75C78.3008 230.065 86.5823 204.625 101.255 183.75C124.255 153.75 158.273 130.25 186.255 130.25Z"
        fill="#FF5D5F"
      />
    </svg>
  );
}

/** Stylized phoenix-flame mark for the Arize Phoenix import surfaces. */
export function PhoenixLogo({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 512 512"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M256 56C256 56 144 140 144 268C144 332 186 388 256 412C326 388 368 332 368 268C368 140 256 56 256 56ZM256 118C282 152 318 206 318 264C318 304 294 338 256 354C218 338 194 304 194 264C194 206 230 152 256 118Z"
        fill="#F97316"
      />
      <path
        d="M256 188C256 188 208 232 208 286C208 318 227 344 256 356C285 344 304 318 304 286C304 232 256 188 256 188Z"
        fill="#FBBF24"
      />
      <path
        d="M118 196C92 232 76 274 76 318C76 396 154 452 256 456C190 432 142 392 124 332C110 286 112 238 118 196Z"
        fill="#F97316"
        opacity="0.55"
      />
      <path
        d="M394 196C420 232 436 274 436 318C436 396 358 452 256 456C322 432 370 392 388 332C402 286 400 238 394 196Z"
        fill="#F97316"
        opacity="0.55"
      />
    </svg>
  );
}

function SetupCommandRow({
  label,
  toastDescription,
  value,
}: {
  label: string;
  toastDescription: string;
  value: string;
}) {
  return (
    <div className="space-y-2">
      <div className="text-xs font-semibold uppercase text-muted-foreground">
        {label}
      </div>
      <CommandBlock
        className="bg-background"
        cmd={value}
        toastDescription={toastDescription}
        wrap={false}
      />
    </div>
  );
}
