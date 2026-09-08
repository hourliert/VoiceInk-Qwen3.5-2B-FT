import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { api, queryString } from "../api";
import type { SampleListItem } from "../types";
import { Badge, Empty, ErrorState, Loading, Page, formatDate } from "../components/ui";

export function SampleListPage({scope}: {scope: "recent" | "history"}) {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") || "";
  const status = params.get("status") || "";
  const request = useQuery({
    queryKey: ["samples", scope, q, status],
    queryFn: ({signal}) => api.get<{items: SampleListItem[]}>(
      "/samples" + queryString({
        scope,
        q: scope === "history" ? q : "",
        status: scope === "history" ? status : "",
        limit: scope === "recent" ? 20 : 100,
      }),
      signal,
    ),
  });
  const title = scope === "recent" ? "Recent VoiceInk use" : "Annotation history";
  return (
    <Page
      title={title}
      eyebrow={scope === "recent" ? "Daily labeling" : "Registry"}
      actions={<button className="button secondary" onClick={() => request.refetch()}>Refresh</button>}
    >
      {scope === "history" && (
        <div className="toolbar">
          <input
            aria-label="Search history"
            placeholder="Search transcript or request ID"
            value={q}
            onChange={(event) => {
              const next = new URLSearchParams(params);
              if (event.target.value) next.set("q", event.target.value);
              else next.delete("q");
              setParams(next, {replace: true});
            }}
          />
          <select
            aria-label="Decision status"
            value={status}
            onChange={(event) => {
              const next = new URLSearchParams(params);
              if (event.target.value) next.set("status", event.target.value);
              else next.delete("status");
              setParams(next, {replace: true});
            }}
          >
            <option value="">All decisions</option>
            <option value="reviewed">Reviewed</option>
            <option value="unreviewed">Unreviewed</option>
          </select>
        </div>
      )}
      {request.isLoading && <Loading label="Loading samples" />}
      {request.error && <ErrorState error={request.error} />}
      {request.data && !request.data.items.length && <Empty>No samples match this view.</Empty>}
      <div className="sample-list">
        {request.data?.items.map((sample) => (
          <Link className="sample-row" to={"/samples/" + encodeURIComponent(sample.request_id)} key={sample.request_id}>
            <div className="sample-row-main">
              <p>{sample.transcript}</p>
              <div className="sample-meta">
                <span>{formatDate(sample.timestamp)}</span>
                <span>{sample.production_model || "Unknown model"}</span>
              </div>
            </div>
            <Badge tone={sample.outcome ? "good" : sample.job_state === "completed" ? "info" : "neutral"}>
              {sample.outcome ? "Reviewed" : sample.job_state === "completed" ? "Luna ready" : "New"}
            </Badge>
          </Link>
        ))}
      </div>
    </Page>
  );
}
