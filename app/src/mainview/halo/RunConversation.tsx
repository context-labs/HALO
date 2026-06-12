import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  Loader2,
  RotateCcw,
  User,
} from "lucide-react";

import { Button, cn } from "~/lib/ui";
import {
  ChatAgentContent,
  ChatAvatar,
  ChatTurn,
  ChatUserBubble,
} from "~/components/chat";
import type {
  HaloRunEvent,
  HaloRunTurn,
} from "../../server/halo/types";
import { OpenInToolBar } from "./OpenInToolBar";
import { RunPhaseTimeline } from "./RunPhaseTimeline";
import { RunReportView } from "./RunReportView";
import { TurnActivityLog } from "./TurnActivityLog";
import type { HaloRunView } from "./runShared";

/** The chat thread: alternating user prompts and HALO analysis turns. */
export function RunConversation({
  events,
  onRetry,
  run,
  streamText,
  turns,
}: {
  events: HaloRunEvent[];
  onRetry: () => void;
  run: HaloRunView;
  streamText: Record<number, string>;
  turns: HaloRunTurn[];
}) {
  const lastAnsweredTurnIndex = turns.reduce(
    (latest, turn) =>
      turn.role === "assistant" &&
      (turn.status === "completed" || turn.status === "incomplete") &&
      turn.content.trim()
        ? turn.turnIndex
        : latest,
    -1,
  );

  return (
    <div className="space-y-7">
      {turns.map((turn) =>
        turn.role === "user" ? (
          <UserTurn key={turn.id} turn={turn} />
        ) : (
          <AssistantTurn
            events={events.filter(
              (event) => (event.turnIndex ?? 1) === turn.turnIndex,
            )}
            key={turn.id}
            onRetry={onRetry}
            run={run}
            showToolBar={turn.turnIndex === lastAnsweredTurnIndex}
            streamedText={streamText[turn.turnIndex] ?? ""}
            turn={turn}
          />
        ),
      )}
    </div>
  );
}

function UserTurn({ turn }: { turn: HaloRunTurn }) {
  return (
    <ChatTurn
      avatar={
        <ChatAvatar className="bg-muted">
          <User className="h-3.5 w-3.5 text-muted-foreground" />
        </ChatAvatar>
      }
      name="You"
      timestamp={turn.createdAt}
    >
      <ChatUserBubble text={turn.content} />
    </ChatTurn>
  );
}

function AssistantTurn({
  events,
  onRetry,
  run,
  showToolBar,
  streamedText,
  turn,
}: {
  events: HaloRunEvent[];
  onRetry: () => void;
  run: HaloRunView;
  showToolBar: boolean;
  streamedText: string;
  turn: HaloRunTurn;
}) {
  const agentSteps = useMemo(
    () => events.filter((event) => event.eventType === "agent_step"),
    [events],
  );

  const inFlight = turn.status === "pending" || turn.status === "streaming";
  const failed = turn.status === "failed" || turn.status === "cancelled";
  const answer = turn.content.trim() ? turn.content : streamedText;

  return (
    <ChatTurn
      avatar={
        <ChatAvatar className="bg-detail-brand/10">
          <BrainCircuit className="h-3.5 w-3.5 text-detail-brand" />
        </ChatAvatar>
      }
      name="HALO"
      timestamp={turn.finishedAt ?? turn.createdAt}
    >
      <ChatAgentContent>
        {/* The response renders identically while streaming and once final,
            so completion doesn't restyle anything. */}
        {failed ? (
          <div className="rounded-xl border border-destructive-border bg-destructive/5 px-4 py-3">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2">
                <AlertCircle className="mt-1 h-4 w-4 shrink-0 text-destructive" />
                <p className="text-sm text-destructive">
                  {turn.status === "cancelled"
                    ? "This turn was cancelled."
                    : (turn.errorMessage ?? "This turn failed.")}
                </p>
              </div>
              <Button onClick={onRetry} size="sm" variant="outline">
                <RotateCcw className="mr-2 h-3.5 w-3.5" />
                Retry
              </Button>
            </div>
            {answer ? (
              <pre className="mt-3 whitespace-pre-wrap font-sans text-sm leading-6 text-muted-foreground">
                {answer}
              </pre>
            ) : null}
          </div>
        ) : answer ? (
          /* Slightly dimmed body so the sender name carries more contrast. */
          <div
            className={cn(
              "min-w-0 text-foreground/85",
              inFlight && "stream-block-fade",
            )}
          >
            <RunReportView markdown={answer} />
          </div>
        ) : inFlight ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            HALO is digging through the traces…
          </p>
        ) : (
          <p className="text-sm italic text-muted-foreground">
            HALO returned no answer for this turn.
          </p>
        )}

        {agentSteps.length > 0 || inFlight ? (
          // pt-2 on top of the container's space-y-3 ≈ 20px gap below the text.
          <div className="pt-2">
            <ActivityDisclosure
              inFlight={inFlight}
              run={run}
              stepCount={agentSteps.length}
              steps={agentSteps}
              turn={turn}
            />
          </div>
        ) : null}

        {/* Hand the report straight to a coding agent from the thread. */}
        {showToolBar && !inFlight ? (
          <OpenInToolBar layout="row" runId={run.id} />
        ) : null}
      </ChatAgentContent>
    </ChatTurn>
  );
}

function ActivityDisclosure({
  inFlight,
  run,
  stepCount,
  steps,
  turn,
}: {
  inFlight: boolean;
  run: HaloRunView;
  stepCount: number;
  steps: HaloRunEvent[];
  turn: HaloRunTurn;
}) {
  // Open while the turn streams, collapse when it lands — the user can still
  // toggle it manually at any point.
  const [open, setOpen] = useState(inFlight);
  useEffect(() => {
    setOpen(inFlight);
  }, [inFlight]);

  const stepsLabel = `${stepCount} agent step${stepCount === 1 ? "" : "s"}`;
  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-subtle",
        open ? "bg-card" : "bg-transparent",
      )}
    >
      <button
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm text-muted-foreground transition hover:text-foreground"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          {open ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0" />
          )}
          {inFlight ? (
            <RunPhaseTimeline run={run} />
          ) : (
            <span>Worked for {workedFor(turn)}</span>
          )}
        </span>
        <span className="shrink-0 text-xs">{stepsLabel}</span>
      </button>
      {open ? (
        <div className="border-t border-subtle">
          <TurnActivityLog events={steps} live={inFlight} />
        </div>
      ) : null}
    </div>
  );
}

function workedFor(turn: HaloRunTurn) {
  const start = Date.parse(turn.createdAt);
  const end = turn.finishedAt ? Date.parse(turn.finishedAt) : Date.now();
  const seconds = Math.max(1, Math.round((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s`;
}
