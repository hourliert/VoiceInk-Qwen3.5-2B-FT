import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { Overview } from "../types";
import { Badge, Card, ErrorState, Loading, Page, Progress } from "../components/ui";

export function OverviewPage() {
  const overview = useQuery({
    queryKey: ["overview"],
    queryFn: ({signal}) => api.get<Overview>("/overview", signal),
  });
  if (overview.isLoading) return <Page title="Overview"><Loading /></Page>;
  if (overview.error || !overview.data) {
    return <Page title="Overview"><ErrorState error={overview.error} /></Page>;
  }
  const data = overview.data;
  const cohort = data.active_cohort;
  return (
    <Page
      title="Good morning, Thomas"
      eyebrow="VoiceInk workspace"
      actions={<button className="button secondary" onClick={() => overview.refetch()}>Refresh</button>}
    >
      <div className="hero-grid">
        <Card className="hero-card">
          <p className="eyebrow">Production model</p>
          <h2>{data.production_model || "No recent model"}</h2>
          <p className="muted">Current model observed by the proxy.</p>
          <Link className="text-link" to="/models">View model inventory →</Link>
        </Card>
        <Card className="hero-card">
          <p className="eyebrow">Registry</p>
          <div className="metric-row">
            <strong>{data.stats.samples.toLocaleString()}</strong>
            <span>samples</span>
          </div>
          <p className="muted">
            {data.stats.reviewed.toLocaleString()} current decisions ·{" "}
            {data.stats.failed_jobs} failed jobs
          </p>
          <Link className="text-link" to="/review/history">Browse history →</Link>
        </Card>
      </div>
      {cohort ? (
        <Card title="Active cohort" className="section-card">
          <div className="card-title-row">
            <div>
              <Link className="resource-title" to={"/data/cohorts/" + encodeURIComponent(cohort.name)}>
                {cohort.name}
              </Link>
              <p className="muted">{cohort.status.members.toLocaleString()} canonical samples</p>
            </div>
            <Badge tone={cohort.status.ready ? "good" : "warn"}>
              {cohort.status.ready ? "Ready" : "Reviewing"}
            </Badge>
          </div>
          <div className="progress-grid">
            <div>
              <p className="label">Fresh Luna analyses</p>
              <Progress value={cohort.status.fresh_luna} total={cohort.status.members} />
            </div>
            <div>
              <p className="label">Required human review</p>
              <Progress
                value={cohort.status.required_human_done}
                total={cohort.status.required_human}
              />
            </div>
          </div>
          <div className="button-row">
            <Link className="button primary" to={"/review/queue?cohort=" + encodeURIComponent(cohort.name) + "&review=required"}>
              Continue review
            </Link>
            <Link className="button secondary" to={"/data/cohorts/" + encodeURIComponent(cohort.name)}>
              Release readiness
            </Link>
          </div>
        </Card>
      ) : (
        <Card title="No active cohort">
          <p className="muted">Create a cohort from the canonical data workflow.</p>
        </Card>
      )}
      <div className="three-grid">
        <Link className="action-card" to="/review/recent">
          <span>Daily workflow</span>
          <strong>Label recent VoiceInk use</strong>
          <p>Inspect production mistakes while the context is fresh.</p>
        </Link>
        <Link className="action-card" to="/runs/training">
          <span>Experiments</span>
          <strong>Monitor training</strong>
          <p>See concise run state, then open full metrics in MLflow.</p>
        </Link>
        <Link className="action-card" to="/runs/evaluations">
          <span>Quality gate</span>
          <strong>Review evaluations</strong>
          <p>Compare candidates on the locked benchmark and acceptance set.</p>
        </Link>
      </div>
    </Page>
  );
}
