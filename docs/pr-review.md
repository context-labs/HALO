TODO:

- [ ] Unit-based args/results should include the unit in the field name (`*_bytes`, `*_chars`, etc.) so tool callers know what budgets mean
- [ ] Search trace returns bounded match records with matched text, surrounding context, lightweight span metadata, and span size. It should not return full span payloads.

SearchTraceArguments(BaseModel):
    trace_id: str
    regex_pattern: str # compile as regex internally; invalid regex should fail clearly
    context_buffer_chars: int = 100
    max_matches: int = 50

SearchTraceResult(BaseModel):
    trace_id: str
    match_count: int
    returned_match_count: int
    has_more: bool
    matches: list[SpanMatchRecord]


SpanMatchRecord(BaseModel):
    trace_id: str
    span_id: str
    span_index: int
    span_name: str
    kind: str
    status_code: str
    parent_span_id: str
    span_serialized_bytes: int
    match_text: str
    matched_context: str
    match_start_char: int
    match_end_char: int

- [ ] ViewSpans should not be called on spans with individual serialized size exceeding 4k bytes or total serialized size exceeding 150k bytes (instructions in tool definition). Also enforce a hard total response budget in code; instructions alone are not enough.
- [ ] SearchSpan tool which is used to match regex inside of a span (similar to search trace but for a single span). It should return a list of matches, not just one match.

SearchSpanArguments(BaseModel):
    trace_id: str
    span_id: str
    regex_pattern: str # compile as regex internally; invalid regex should fail clearly
    context_buffer_chars: int = 100
    max_matches: int = 50

SearchSpanResult(BaseModel):
    trace_id: str
    span_id: str
    match_count: int
    returned_match_count: int
    has_more: bool
    matches: list[SpanMatchRecord]


- [ ] update tool instructions to encourage desired flow
- [ ] remove bubblewrap from uv.lock
- [ ] Query traces should return the total serialized size of the trace so that we know if we can call view_trace or not

class TraceSummary(BaseModel):
    """Slim per-trace projection used in query results — purely from the index, no JSONL reads."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_count: int = Field(ge=0)
    start_time: str
    end_time: str
    has_errors: bool
    service_names: list[str]
    model_names: list[str]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    agent_names: list[str]
    total_serialized_bytes: int = Field(ge=0)

- [ ] query traces should have a regex option. Prefer making this explicit on QueryTracesArguments as `content_regex_pattern` rather than hiding scan-heavy behavior inside the common TraceFilters used by overview/count. Normal filters stay index-only; `content_regex_pattern` scans raw span JSON for matching traces.

class TraceFilters(BaseModel):
    """Common filter set applied across overview/query/count. All fields are optional ANDed predicates."""

    model_config = ConfigDict(extra="forbid")

    has_errors: bool | None = None
    model_names: list[str] | None = None
    service_names: list[str] | None = None
    agent_names: list[str] | None = None
    project_id: str | None = None
    start_time_gte: str | None = None
    end_time_lte: str | None = None

class QueryTracesArguments(BaseModel):
    filters: TraceFilters = Field(default_factory=TraceFilters)
    content_regex_pattern: str | None = None # compile as regex internally
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

- [ ] repo readme
- [ ] demo project readme
- [ ] reproduce results Amar achieved


1. looking for some pattern in the dataset
    - call query traces and get a set of trace ids 
        - ideally we also get stats on each trace to know if we should call view_trace or not
            - if we call view trace and we get truncated spans, we need to call run code to surgically inspect those spans
2. looking at a specific trace
    - if trace is short enough, call view_trace
    - if trace is too long, we need to call search trace to get spans ids, which should also give us span sizes
        - if spans are short, call view_spans
        - if spans are too long, we need to call run_code to surgically inspect or search span???


Render / synthesis path (keep separate from span viewing changes):

- [ ] Remove `synthesize_traces`
- [ ] Remove `render_trace`
- [ ] Prefer subagent-driven synthesis: the root agent should delegate focused trace-gathering or summarization tasks to a subagent, and that subagent should use `query_traces`, `search_trace`, `search_span`, and `view_spans` directly
- [ ] Add root/subagent instructions for when to delegate synthesis work, e.g. when comparing multiple traces, gathering evidence from large traces, or producing a concise summary from several search results
- [ ] Subagent summaries should cite concrete trace ids, span ids, status/error patterns, model/tool names, and relevant token/count metadata from the trace tools
- [ ] Do not keep a dedicated synthesis tool; it duplicates the interactive trace primitives and risks hidden lossy rendering
