import {
  Activity,
  Filter,
  MessageSquare,
} from "lucide-react";

import { Button, Tabs, TabsList, TabsTrigger } from "~/lib/ui";
import { FilterSelect } from "~/components/FilterSelect";
import { toFacetOptions, sourceLabel } from "~/lib/format";
import type { FacetOption } from "../../server/telemetry/types";
import type {
  ScopeFilter,
  SourceFilter,
  StatusFilter,
  TraceMonitorViewMode,
} from "./filters";

export function FilterSidebar({
  agentName,
  description,
  facets,
  modelName,
  onAgentNameChange,
  onModelNameChange,
  onReset,
  onScopeChange,
  onServiceNameChange,
  onSourceChange,
  onStatusChange,
  onViewModeChange,
  scope,
  serviceName,
  source,
  status,
  viewMode,
}: {
  agentName: string;
  description: string;
  facets: Partial<Record<string, FacetOption[]>>;
  modelName: string;
  onAgentNameChange: (value: string) => void;
  onModelNameChange: (value: string) => void;
  onReset: () => void;
  onScopeChange: (value: ScopeFilter) => void;
  onServiceNameChange: (value: string) => void;
  onSourceChange: (value: SourceFilter) => void;
  onStatusChange: (value: StatusFilter) => void;
  onViewModeChange?: (value: TraceMonitorViewMode) => void;
  scope: ScopeFilter;
  serviceName: string;
  source: SourceFilter;
  status: StatusFilter;
  viewMode?: TraceMonitorViewMode;
}) {
  return (
    <aside className="border-r border-subtle bg-sidebar">
      <div className="flex h-full flex-col gap-5 p-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Filter className="h-4 w-4" />
            Filters
          </div>
          <p className="text-xs text-muted-foreground">
            {description}
          </p>
        </div>

        <div className="space-y-4">
          {viewMode && onViewModeChange ? (
            <div className="space-y-2">
              <span className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground">
                View
              </span>
              <Tabs
                onValueChange={(value) => {
                  if (value === "traces" || value === "sessions") {
                    onViewModeChange(value);
                  }
                }}
                value={viewMode}
              >
                <TabsList className="grid w-full grid-cols-2 gap-1 rounded-md border border-subtle bg-background-muted p-1 sm:grid sm:w-full">
                  <TabsTrigger
                    className="w-full gap-1.5 px-2 py-1.5 text-xs"
                    value="traces"
                  >
                    <Activity className="h-3.5 w-3.5" />
                    Traces
                  </TabsTrigger>
                  <TabsTrigger
                    className="w-full gap-1.5 px-2 py-1.5 text-xs"
                    value="sessions"
                  >
                    <MessageSquare className="h-3.5 w-3.5" />
                    Sessions
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          ) : null}
          <FilterSelect
            label="Status"
            onChange={(value) => onStatusChange(value as StatusFilter)}
            options={[
              { label: "Any status", value: "all" },
              { label: "OK", value: "ok" },
              { label: "Errors", value: "error" },
            ]}
            value={status}
          />
          <FilterSelect
            label="Scope"
            onChange={(value) => onScopeChange(value as ScopeFilter)}
            options={[
              { label: "All spans", value: "all" },
              { label: "Root spans", value: "root" },
              { label: "Entrypoints", value: "entrypoint" },
            ]}
            value={scope}
          />
          <FilterSelect
            label="Source"
            onChange={(value) => onSourceChange(value as SourceFilter)}
            options={toFacetOptions(facets.source, "Any source").map((option) => ({
              ...option,
              label: sourceLabel(option.value, option.label),
            }))}
            value={source}
          />
          <FilterSelect
            label="Service"
            onChange={onServiceNameChange}
            options={toFacetOptions(facets.service_name, "Any service")}
            value={serviceName}
          />
          <FilterSelect
            label="Agent"
            onChange={onAgentNameChange}
            options={toFacetOptions(facets.agent_name, "Any agent")}
            value={agentName}
          />
          <FilterSelect
            label="Model"
            onChange={onModelNameChange}
            options={toFacetOptions(facets.llm_model_name, "Any model")}
            value={modelName}
          />
        </div>

        <Button className="mt-auto" onClick={onReset} variant="outline">
          Reset filters
        </Button>
      </div>
    </aside>
  );
}
