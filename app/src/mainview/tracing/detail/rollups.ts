import type { SpanNode } from "../../../server/telemetry/types";

/** Sum of costTotal across a node and its descendants; null when nothing contributed. */
export function rollupCost(node: SpanNode): number | null {
  let total: number | null = null;
  const visit = (current: SpanNode) => {
    const cost = current.span.costTotal;
    if (cost != null) {
      const value = Number(cost);
      if (Number.isFinite(value)) total = (total ?? 0) + value;
    }
    current.children.forEach(visit);
  };
  // Synthetic session-group spans copy trace cost — counting children too
  // would double it, so only descend.
  if (node.span.spanId.startsWith("session:")) {
    const cost = node.span.costTotal == null ? null : Number(node.span.costTotal);
    return cost != null && Number.isFinite(cost) ? cost : null;
  }
  visit(node);
  return total;
}

/** [in, out] token rollup; nulls when nothing contributed. */
export function rollupTokens(node: SpanNode): [number | null, number | null] {
  let input: number | null = null;
  let output: number | null = null;
  const visit = (current: SpanNode) => {
    if (current.span.inputTokens != null) {
      input = (input ?? 0) + current.span.inputTokens;
    }
    if (current.span.outputTokens != null) {
      output = (output ?? 0) + current.span.outputTokens;
    }
    current.children.forEach(visit);
  };
  visit(node);
  return [input, output];
}
