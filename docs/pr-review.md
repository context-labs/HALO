TODO:

- [ ] Search trace retuns span ids with matched text, the span size, and the matched text with surrounding context

SearchTraceArguments(BaseModel):
    trace_id: str
    pattern: re.Regex
    context_buffer_char_size: int = 100

SearchTraceResult(BaseModel):
    matched_spans: list[SpanSummaryRecord]


SpanSummaryRecord(BaseModel):
    span_id: str
    span_size: int
    match: str
    matched_context: str
    match_start_index: int
    match_end_index: int

- [ ] ViewSpans should not be called on spans with individual size exceeding 4k or total size exceeding 150k (instructions in tool definition)
- [ ] SearchSpan tool which is used to match regex inside of a span (similar to search trace but for a single span)

SearchSpanArguments(BaseModel):
    trace_id: str
    span_id: str
    pattern: re.Regex
    context_buffer_char_size: int = 100

SearchSpanResult(BaseModel):
    trace_id: str
    span_id: str
    match: str
    matched_context: str
    match_start_index: int
    match_end_index: int


- [ ] update tool instructions to encourage desired flow
- [ ] remove bubblewrap from uv.lock
- [ ] Query traces should return the total size of the trace so that we know if we can call view_trace or not

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
    total_serialized_chars: int = Field(ge=0) # or similar

- [ ] query traces should have a regex option

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
    pattern: re.Regex | None = None

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
