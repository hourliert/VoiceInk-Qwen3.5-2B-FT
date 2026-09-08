import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { RunSummary } from "../types";
import { Badge, Card, Empty, ErrorState, Loading, Page, formatDate } from "../components/ui";

export function ModelsPage() {
  const query = useQuery({
    queryKey: ["models"],
    queryFn: ({signal}) => api.get<{items: Array<Record<string, unknown>>}>("/models", signal),
  });
  return (
    <Page title="Model inventory" eyebrow="Canonical and experimental artifacts">
      {query.isLoading && <Loading />}
      {query.error && <ErrorState error={query.error} />}
      <div className="resource-list">
        {query.data?.items.map((model) => (
          <Link className="resource-row" to={"/models/" + encodeURIComponent(String(model.name))} key={String(model.name)}>
            <div>
              <strong>{String(model.name)}</strong>
              <p>{String(model.path)} · {(Number(model.bytes) / 1_000_000_000).toFixed(2)} GB</p>
            </div>
            <Badge tone={model.state === "canonical" ? "good" : model.state === "legacy" ? "neutral" : "info"}>
              {String(model.state)}
            </Badge>
          </Link>
        ))}
      </div>
    </Page>
  );
}

export function ModelPage() {
  const {name = ""} = useParams();
  const modelName = decodeURIComponent(name);
  const query = useQuery({
    queryKey: ["model", modelName],
    queryFn: ({signal}) => api.get<Record<string, unknown>>("/models/" + encodeURIComponent(modelName), signal),
  });
  if (query.isLoading) return <Page title={modelName}><Loading /></Page>;
  if (query.error || !query.data) return <Page title={modelName}><ErrorState error={query.error} /></Page>;
  return (
    <Page title={modelName} eyebrow="Model artifact">
      <Card>
        <dl className="detail-list">
          <div><dt>State</dt><dd>{String(query.data.state)}</dd></div>
          <div><dt>Path</dt><dd><code>{String(query.data.path)}</code></dd></div>
          <div><dt>Context</dt><dd>{String(query.data.context)}</dd></div>
          <div><dt>Loaded on startup</dt><dd>{query.data.load_on_startup ? "Yes" : "No"}</dd></div>
        </dl>
      </Card>
    </Page>
  );
}

export function RunsPage({category}: {category: "training" | "evaluations"}) {
  const query = useQuery({
    queryKey: ["runs", category],
    queryFn: ({signal}) => api.get<{items: RunSummary[]; errors: string[]; mlflow_url: string}>(
      "/runs?category=" + category + "&limit=50",
      signal,
    ),
  });
  return (
    <Page
      title={category === "training" ? "Training runs" : "Evaluation runs"}
      eyebrow="MLflow experiment index"
      actions={query.data?.mlflow_url ? <a className="button secondary" href={query.data.mlflow_url} target="_blank" rel="noreferrer">Open MLflow ↗</a> : null}
    >
      {query.isLoading && <Loading label="Loading MLflow summaries" />}
      {query.error && <ErrorState error={query.error} />}
      {query.data?.errors.map((error) => <div className="warning-line" key={error}>{error}</div>)}
      {!query.data?.items.length && !query.isLoading && <Empty>No matching MLflow runs.</Empty>}
      <div className="resource-list">
        {query.data?.items.map((run) => (
          <Link className="resource-row run-row" to={"/runs/" + run.run_id} key={run.run_id}>
            <div>
              <strong>{run.name}</strong>
              <p>{formatDate(run.start_time)} · {run.release || run.profile || run.kind || run.experiment}</p>
            </div>
            <div className="run-summary-side">
              {Object.entries(run.metrics).slice(0, 2).map(([key, value]) => (
                <span key={key}>{key.replaceAll("_", " ")} <strong>{Number(value).toFixed(3)}</strong></span>
              ))}
              <Badge tone={run.status === "FINISHED" ? "good" : run.status === "FAILED" ? "bad" : "info"}>
                {run.status}
              </Badge>
            </div>
          </Link>
        ))}
      </div>
    </Page>
  );
}

export function RunPage() {
  const {runId = ""} = useParams();
  const query = useQuery({
    queryKey: ["run", runId],
    queryFn: ({signal}) => api.get<Record<string, unknown>>("/runs/" + encodeURIComponent(runId), signal),
  });
  if (query.isLoading) return <Page title="MLflow run"><Loading /></Page>;
  if (query.error || !query.data) return <Page title="MLflow run"><ErrorState error={query.error} /></Page>;
  return (
    <Page
      title={String(query.data.name || runId)}
      eyebrow={String(query.data.experiment || "MLflow run")}
      actions={<a className="button primary" href={String(query.data.mlflow_url)} target="_blank" rel="noreferrer">Open full MLflow run ↗</a>}
    >
      <Card title="Metrics"><pre className="json-view">{JSON.stringify(query.data.metrics || {}, null, 2)}</pre></Card>
      <Card title="Parameters"><pre className="json-view">{JSON.stringify(query.data.params || {}, null, 2)}</pre></Card>
      <Card title="Tags"><pre className="json-view">{JSON.stringify(query.data.tags || {}, null, 2)}</pre></Card>
    </Page>
  );
}

export function SystemPage() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({signal}) => api.get<Record<string, unknown>>("/health", signal),
  });
  const system = useQuery({
    queryKey: ["system"],
    queryFn: ({signal}) => api.get<Record<string, unknown>>("/system", signal),
  });
  return (
    <Page
      title="System"
      eyebrow="LAN services and workers"
      actions={<button className="button secondary" onClick={() => {
        health.refetch();
        system.refetch();
      }}>Refresh checks</button>}
    >
      {(health.isLoading || system.isLoading) && <Loading />}
      {(health.error || system.error) && <ErrorState error={health.error || system.error} />}
      {health.data && (
        <Card title="Control plane">
          <pre className="json-view">{JSON.stringify(health.data, null, 2)}</pre>
        </Card>
      )}
      {system.data && (
        <Card title="Services and GPU">
          <pre className="json-view">{JSON.stringify(system.data, null, 2)}</pre>
        </Card>
      )}
    </Page>
  );
}

export function DocsPage() {
  const canonical = useQuery({
    queryKey: ["docs", "canonical"],
    queryFn: ({signal}) => api.get<string>("/docs/canonical", signal),
  });
  const review = useQuery({
    queryKey: ["docs", "review"],
    queryFn: ({signal}) => api.get<string>("/docs/review", signal),
  });
  return (
    <Page title="Operating documentation" eyebrow="Architecture and workflows">
      {(canonical.isLoading || review.isLoading) && <Loading />}
      {(canonical.error || review.error) && <ErrorState error={canonical.error || review.error} />}
      <Card title="Canonical pipeline"><pre className="docs-view">{canonical.data}</pre></Card>
      <Card title="Daily labeling"><pre className="docs-view">{review.data}</pre></Card>
    </Page>
  );
}
