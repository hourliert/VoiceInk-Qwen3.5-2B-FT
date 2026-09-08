import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { Cohort, JsonObject, ReleasePreview } from "../types";
import { Badge, Card, Empty, ErrorState, Loading, Page, Progress, formatDate } from "../components/ui";

export function CohortsPage() {
  const query = useQuery({
    queryKey: ["cohorts"],
    queryFn: ({signal}) => api.get<{items: Cohort[]}>("/cohorts", signal),
  });
  return (
    <Page title="Canonical cohorts" eyebrow="Dataset lifecycle">
      {query.isLoading && <Loading />}
      {query.error && <ErrorState error={query.error} />}
      {!query.data?.items.length && !query.isLoading && <Empty>No cohorts yet.</Empty>}
      <div className="resource-list">
        {query.data?.items.map((cohort) => (
          <Link className="resource-row" to={"/data/cohorts/" + encodeURIComponent(cohort.name)} key={cohort.name}>
            <div>
              <strong>{cohort.name}</strong>
              <p>{cohort.status.members} samples · created {formatDate(cohort.created_at)}</p>
            </div>
            <Badge tone={cohort.status.ready ? "good" : "warn"}>{cohort.status.state}</Badge>
          </Link>
        ))}
      </div>
    </Page>
  );
}

export function CohortPage() {
  const {name = ""} = useParams();
  const cohortName = decodeURIComponent(name);
  const queryClient = useQueryClient();
  const [releaseName, setReleaseName] = useState("");
  const query = useQuery({
    queryKey: ["cohort", cohortName],
    queryFn: ({signal}) => api.get<Cohort>("/cohorts/" + encodeURIComponent(cohortName), signal),
  });
  const preview = useQuery({
    queryKey: ["release-preview", cohortName],
    queryFn: ({signal}) => api.get<ReleasePreview>(
      "/cohorts/" + encodeURIComponent(cohortName) + "/release-preview", signal,
    ),
  });
  const release = useMutation({
    mutationFn: () => api.post<JsonObject>(
      "/cohorts/" + encodeURIComponent(cohortName) + "/releases",
      {name: releaseName, correction_cutoff: preview.data?.correction_cutoff,
       selection_sha256: preview.data?.selection_sha256},
    ),
    onSuccess: () => {
      queryClient.invalidateQueries({queryKey: ["cohort", cohortName]});
      queryClient.invalidateQueries({queryKey: ["releases"]});
    },
  });
  if (query.isLoading) return <Page title={cohortName}><Loading /></Page>;
  if (query.error || !query.data) return <Page title={cohortName}><ErrorState error={query.error} /></Page>;
  const status = query.data.status;
  return (
    <Page
      title={cohortName}
      eyebrow="Canonical cohort"
      actions={<Link className="button primary" to={"/review/queue?cohort=" + encodeURIComponent(cohortName) + "&review=required"}>Open review queue</Link>}
    >
      <div className="hero-grid">
        <Card title="Luna coverage">
          <Progress value={status.fresh_luna} total={status.members} />
        </Card>
        <Card title="Human review">
          <Progress value={status.required_human_done} total={status.required_human} />
        </Card>
      </div>
      <Card title="Release readiness">
        <div className="readiness-grid">
          <div><span>Members</span><strong>{status.members} / 1,200</strong></div>
          <div><span>Unresolved critical</span><strong>{status.unresolved_critical}</strong></div>
          <div><span>Acceptance</span><strong>{status.splits.acceptance || 0} / 150</strong></div>
          <div><span>State</span><Badge tone={status.ready ? "good" : "warn"}>{status.state}</Badge></div>
        </div>
        {status.expanded_review_strata.length > 0 && (
          <p className="warning-line">Expanded audit: {status.expanded_review_strata.join(", ")}</p>
        )}
        {preview.isLoading && <Loading />}
        {preview.error && <ErrorState error={preview.error} />}
        {preview.data && (
          <div className="readiness-grid release-preview">
            <div><span>Training</span><strong>{preview.data.splits.train.toLocaleString()}</strong></div>
            <div><span>Validation</span><strong>{preview.data.splits.validation}</strong></div>
            <div><span>Acceptance</span><strong>{preview.data.splits.acceptance}</strong></div>
            <div><span>Extra human corrections</span><strong>{preview.data.additions.length}</strong></div>
          </div>
        )}
        {preview.data && <p className="muted">The selection above is frozen by SHA-256 when you create the release. Extra corrections are train-only; validation, acceptance, and the locked 440 remain uncontaminated.</p>}
        <div className="release-form">
          <input
            value={releaseName}
            placeholder="Release name, e.g. voiceink-2026-09-v1"
            onChange={(event) => setReleaseName(event.target.value)}
          />
          <button
            className="button primary"
            disabled={!status.ready || !releaseName.trim() || !preview.data || release.isPending}
            onClick={() => release.mutate()}
          >
            {release.isPending ? "Creating…" : "Create immutable release"}
          </button>
        </div>
        {!status.ready && <p className="muted">Release creation stays locked until all fail-closed checks pass.</p>}
        {release.error && <ErrorState error={release.error} />}
        {release.data && <p className="success-line">Release created successfully. Open Immutable releases for its manifest and training commands.</p>}
      </Card>
    </Page>
  );
}

export function ReleasesPage() {
  const query = useQuery({
    queryKey: ["releases"],
    queryFn: ({signal}) => api.get<{items: Array<Record<string, unknown>>}>("/releases", signal),
  });
  return (
    <Page title="Immutable releases" eyebrow="Dataset lifecycle">
      {query.isLoading && <Loading />}
      {query.error && <ErrorState error={query.error} />}
      {!query.data?.items.length && !query.isLoading && (
        <Empty>No canonical release has been sealed yet.</Empty>
      )}
      <div className="resource-list">
        {query.data?.items.map((release) => (
          <Link className="resource-row" to={"/data/releases/" + encodeURIComponent(String(release.name))} key={String(release.name)}>
            <div>
              <strong>{String(release.name)}</strong>
              <p>{Number(release.records).toLocaleString()} records · {String(release.cohort_name)}</p>
            </div>
            <Badge tone={release.sealed ? "good" : "warn"}>{release.sealed ? "sealed" : "draft"}</Badge>
          </Link>
        ))}
      </div>
    </Page>
  );
}

export function ReleasePage() {
  const {name = ""} = useParams();
  const releaseName = decodeURIComponent(name);
  const query = useQuery({
    queryKey: ["release", releaseName],
    queryFn: ({signal}) => api.get<Record<string, unknown>>("/releases/" + encodeURIComponent(releaseName), signal),
  });
  if (query.isLoading) return <Page title={releaseName}><Loading /></Page>;
  if (query.error || !query.data) return <Page title={releaseName}><ErrorState error={query.error} /></Page>;
  const manifestPath = String(query.data.manifest_path || "");
  const preflight = `.venv/bin/python3 src/training/train.py sft --profile qwen35-2b-sft --release-manifest ${manifestPath} --check-only --print-command`;
  const train = `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python3 src/training/train.py sft --profile qwen35-2b-sft --release-manifest ${manifestPath} --export-gguf q4_k_m q8_0`;
  return (
    <Page title={releaseName} eyebrow="Immutable dataset release">
      <Card title="Lineage">
        <dl className="detail-list">
          <div><dt>Cohort</dt><dd>{String(query.data.cohort_name || "")}</dd></div>
          <div><dt>Records</dt><dd>{String(query.data.records || "")}</dd></div>
          <div><dt>Manifest SHA-256</dt><dd><code>{String(query.data.manifest_sha256 || "")}</code></dd></div>
          <div><dt>Manifest</dt><dd><code>{String(query.data.manifest_path || "")}</code></dd></div>
        </dl>
      </Card>
      <Card title="Manifest snapshot">
        <pre className="json-view">{JSON.stringify(query.data.manifest || {}, null, 2)}</pre>
      </Card>
      <Card title="Train Qwen3.5 2B from this release">
        <p className="muted">Run the fail-closed preflight first. Training records this release and profile fingerprint in MLflow.</p>
        <pre className="json-view">{preflight}</pre>
        <pre className="json-view">{train}</pre>
      </Card>
    </Page>
  );
}
