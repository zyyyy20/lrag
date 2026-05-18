import type { ToolResult, WebSource } from "./types";

const WEB_SOURCE_TOOLS = new Set([
  "tavily_search",
  "tavily_extract",
  "tavily_research",
]);

export function webSourcesFromToolResult(result: ToolResult): WebSource[] {
  if (!WEB_SOURCE_TOOLS.has(result.tool)) return [];
  const rawResults = result.data?.results;
  const items = Array.isArray(rawResults)
    ? rawResults
    : typeof result.data?.url === "string"
      ? [result.data]
      : [];
  const seen = new Set<string>();
  const sources: WebSource[] = [];
  for (const raw of items) {
    if (!raw || typeof raw !== "object") continue;
    const item = raw as Record<string, unknown>;
    const url = typeof item.url === "string" ? item.url : "";
    if (!url || seen.has(url)) continue;
    sources.push({
      title: typeof item.title === "string" && item.title ? item.title : url,
      url,
      content_preview:
        typeof item.content === "string"
          ? item.content
          : typeof item.raw_content === "string"
            ? item.raw_content
            : null,
      score: typeof item.score === "number" ? item.score : null,
      source_tool: result.tool,
    });
    seen.add(url);
  }
  return sources;
}

export function mergeWebSources(
  current: WebSource[] | null | undefined,
  incoming: WebSource[]
): WebSource[] {
  const merged = [...(current ?? [])];
  const seen = new Set(merged.map((source) => source.url));
  for (const source of incoming) {
    if (seen.has(source.url)) continue;
    merged.push(source);
    seen.add(source.url);
  }
  return merged;
}
