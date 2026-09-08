import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { api, queryString } from "../api";
import type { Cohort, QueueResponse } from "../types";
import { Badge, Empty, ErrorState, Loading, Page, Progress, formatDate } from "../components/ui";

const reviewOptions = [
  ["required", "Human required"],
  ["unanalyzed", "Missing Luna"],
  ["failed", "Failed analysis"],
  ["reviewed", "Human reviewed"],
  ["all", "All members"],
];

export function QueuePage() {
  const [params, setParams] = useSearchParams();
  const queryClient = useQueryClient();
  const active = useQuery({
    queryKey: ["active-cohort"],
    queryFn: ({signal}) => api.get<Omit<Cohort, "status">>("/active-cohort", signal),
  });
  const cohortName = params.get("cohort") || active.data?.name || "";
  const review = params.get("review") || "required";
  const split = params.get("split") || "";
  const stratum = params.get("stratum") || "";
  const q = params.get("q") || "";
  const cohort = useQuery({
    queryKey: ["cohort", cohortName],
    enabled: Boolean(cohortName),
    queryFn: ({signal}) => api.get<Cohort>("/cohorts/" + encodeURIComponent(cohortName), signal),
  });
  const queue = useQuery({
    queryKey: ["cohort-queue", cohortName, review, split, stratum, q],
    enabled: Boolean(cohortName),
    queryFn: ({signal}) => api.get<QueueResponse>(
      "/cohorts/" + encodeURIComponent(cohortName) + "/queue" + queryString({
        review, split, stratum, q, limit: 200, offset: 0,
      }),
      signal,
    ),
  });
  const queueMissing = useMutation({
    mutationFn: () => api.post<{queued: number; completed: number}>(
      "/cohorts/" + encodeURIComponent(cohortName) + "/analysis-jobs",
      {},
    ),
    onSuccess: () => {
      queryClient.invalidateQueries({queryKey: ["cohort-queue", cohortName]});
      queryClient.invalidateQueries({queryKey: ["cohort", cohortName]});
    },
  });
  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, {replace: true});
  }
  const detailQuery = queryString({cohort: cohortName, review, split, stratum, q});
  return (
    <Page
      title="Review queue"
      eyebrow={cohortName ? "Active cohort · " + cohortName : "Canonical labeling"}
      actions={
        <div className="button-row">
          <button className="button secondary" onClick={() => {
            queue.refetch();
            cohort.refetch();
          }}>Refresh</button>
          <button
            className="button primary"
            disabled={!cohortName || queueMissing.isPending}
            onClick={() => queueMissing.mutate()}
          >
            {queueMissing.isPending ? "Queuing…" : "Queue missing Luna"}
          </button>
        </div>
      }
    >
      {cohort.data && (
        <div className="progress-grid compact-progress">
          <div>
            <p className="label">Fresh Luna analyses</p>
            <Progress value={cohort.data.status.fresh_luna} total={cohort.data.status.members} />
          </div>
          <div>
            <p className="label">Required human review</p>
            <Progress
              value={cohort.data.status.required_human_done}
              total={cohort.data.status.required_human}
            />
          </div>
        </div>
      )}
      <div className="toolbar queue-toolbar">
        <select value={review} onChange={(event) => update("review", event.target.value)}>
          {reviewOptions.map(([value, label]) => <option value={value} key={value}>{label}</option>)}
        </select>
        <select value={split} onChange={(event) => update("split", event.target.value)}>
          <option value="">All splits</option>
          <option value="train">Train</option>
          <option value="validation">Validation</option>
          <option value="acceptance">Acceptance</option>
        </select>
        <input
          placeholder="Search this cohort"
          value={q}
          onChange={(event) => update("q", event.target.value)}
        />
      </div>
      {queueMissing.error && <ErrorState error={queueMissing.error} />}
      {(active.isLoading || queue.isLoading) && <Loading label="Loading review queue" />}
      {(active.error || queue.error) && <ErrorState error={active.error || queue.error} />}
      {queue.data && (
        <p className="result-count">{queue.data.total.toLocaleString()} matching samples</p>
      )}
      {queue.data && !queue.data.items.length && (
        <Empty>
          {review === "required"
            ? "No Luna-ready samples currently require human review."
            : "No samples match these filters."}
        </Empty>
      )}
      <div className="sample-list">
        {queue.data?.items.map((item) => (
          <Link
            className="sample-row"
            to={"/samples/" + encodeURIComponent(item.request_id) + detailQuery}
            key={item.request_id}
          >
            <div className="sample-row-main">
              <p>{item.transcript}</p>
              <div className="sample-meta">
                <span>{formatDate(item.timestamp)}</span>
                <span>{item.split} · {item.stratum}</span>
                {item.audit_selected ? <span>audit</span> : null}
              </div>
            </div>
            <div className="badge-stack">
              {item.job_state === "failed" && <Badge tone="bad">Failed</Badge>}
              {item.review_pending && <Badge tone="warn">Review</Badge>}
              {item.human_reviewed && <Badge tone="good">Done</Badge>}
              {!item.fresh_luna && <Badge>Waiting for Luna</Badge>}
            </div>
          </Link>
        ))}
      </div>
    </Page>
  );
}
