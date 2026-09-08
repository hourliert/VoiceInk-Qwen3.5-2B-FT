import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, queryString } from "../api";
import type { Cohort, QueueResponse, Sample } from "../types";
import {
  Badge,
  Card,
  DiffText,
  ErrorState,
  Loading,
  Page,
  ScoreGrid,
  formatDate,
} from "../components/ui";

type DecisionOutcome =
  | "accept_luna"
  | "human_edit"
  | "production_correct"
  | "exclude";

interface DecisionResult {
  ok: boolean;
  decision_id: number;
  annotation_id: number | null;
  maintenance: string;
}

export function SamplePage() {
  const {requestId = ""} = useParams();
  const id = decodeURIComponent(requestId);
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [editor, setEditor] = useState("");
  const [editorDirty, setEditorDirty] = useState(false);
  const [contextDependency, setContextDependency] = useState("unknown");
  const analysisRequested = useRef("");

  const cohortFromUrl = params.get("cohort") || "";
  const active = useQuery({
    queryKey: ["active-cohort"],
    enabled: !cohortFromUrl,
    queryFn: ({signal}) => api.get<Omit<Cohort, "status">>("/active-cohort", signal),
    retry: false,
  });
  const cohortName = cohortFromUrl || active.data?.name || "";

  const sample = useQuery({
    queryKey: ["sample", id],
    enabled: Boolean(id),
    queryFn: ({signal}) => api.get<Sample>("/samples/" + encodeURIComponent(id), signal),
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.job && ["pending", "running"].includes(data.job.state)
        ? 2_000
        : false;
    },
  });

  const latestAnalysis = sample.data?.analyses.at(-1);
  const proposal = useMemo(() => {
    if (!sample.data || !latestAnalysis) return undefined;
    return sample.data.annotations.find(
      (annotation) => annotation.id === latestAnalysis.annotation_id,
    );
  }, [sample.data, latestAnalysis]);

  useEffect(() => {
    setEditorDirty(false);
    setEditor("");
    setContextDependency("unknown");
    analysisRequested.current = "";
  }, [id]);

  useEffect(() => {
    if (!sample.data || editorDirty || editor) return;
    setEditor(proposal?.text || sample.data.production_output || "");
  }, [sample.data, proposal, editor, editorDirty]);

  useEffect(() => {
    if (!sample.data || proposal || sample.data.job?.state === "running"
        || sample.data.job?.state === "pending" || analysisRequested.current === id) {
      return;
    }
    analysisRequested.current = id;
    api.post(
      "/samples/" + encodeURIComponent(id) + "/analysis-jobs",
      cohortName ? {cohort: cohortName} : {},
    ).then(() => {
      queryClient.invalidateQueries({queryKey: ["sample", id]});
    }).catch(() => {
      analysisRequested.current = "";
    });
  }, [cohortName, id, proposal, queryClient, sample.data]);

  const retryAnalysis = useMutation({
    mutationFn: () => api.post(
      "/samples/" + encodeURIComponent(id) + "/analysis-jobs",
      {retry: true, ...(cohortName ? {cohort: cohortName} : {})},
    ),
    onSuccess: () => queryClient.invalidateQueries({queryKey: ["sample", id]}),
  });

  async function navigateAfterSave() {
    await Promise.all([
      queryClient.invalidateQueries({queryKey: ["sample", id]}),
      queryClient.invalidateQueries({queryKey: ["cohort", cohortName]}),
      queryClient.invalidateQueries({queryKey: ["cohort-queue", cohortName]}),
      queryClient.invalidateQueries({queryKey: ["overview"]}),
    ]);
    if (!cohortName) {
      navigate("/review/recent");
      return;
    }
    const split = params.get("split") || "";
    const stratum = params.get("stratum") || "";
    const q = params.get("q") || "";
    const queue = await api.get<QueueResponse>(
      "/cohorts/" + encodeURIComponent(cohortName) + "/queue" + queryString({
        review: "required", split, stratum, q, limit: 500, offset: 0,
      }),
    );
    const next = queue.items.find((item) => (
      item.request_id !== id && item.fresh_luna && item.review_pending
    ));
    if (next) {
      navigate(
        "/samples/" + encodeURIComponent(next.request_id) + queryString({
          cohort: cohortName, review: "required", split, stratum, q,
        }),
        {replace: true},
      );
    } else {
      navigate(
        "/review/queue" + queryString({
          cohort: cohortName, review: "required", split, stratum, q,
        }),
        {replace: true},
      );
    }
  }

  const decision = useMutation({
    mutationFn: (outcome: DecisionOutcome) => api.post<DecisionResult>(
      "/samples/" + encodeURIComponent(id) + "/decisions",
      {
        outcome,
        context_dependency: contextDependency,
        ...(outcome === "human_edit" ? {label: editor} : {}),
      },
    ),
    onSuccess: navigateAfterSave,
  });

  const backHref = cohortName
    ? "/review/queue" + queryString({
        cohort: cohortName,
        review: params.get("review") || "required",
        split: params.get("split") || "",
        stratum: params.get("stratum") || "",
        q: params.get("q") || "",
      })
    : "/review/recent";

  if (sample.isLoading) {
    return <Page title="Sample"><Loading label="Loading sample" /></Page>;
  }
  if (sample.error || !sample.data) {
    return <Page title="Sample"><ErrorState error={sample.error} /></Page>;
  }
  const data = sample.data;
  const analysis = latestAnalysis;
  const jobBusy = data.job && ["pending", "running"].includes(data.job.state);
  return (
    <Page
      title="Review sample"
      eyebrow={formatDate(data.timestamp)}
      actions={<Link className="button secondary" to={backHref}>← Back to queue</Link>}
    >
      <div className="sample-heading">
        <div className="badge-stack horizontal">
          <Badge tone={analysis ? "info" : "neutral"}>
            {analysis ? "Luna ready" : jobBusy ? "Luna running" : "Waiting for Luna"}
          </Badge>
          {analysis && <Badge tone={analysis.confidence === "high" ? "good" : "warn"}>
            {analysis.confidence} confidence
          </Badge>}
          {analysis?.material_difference && <Badge tone="warn">Material edit</Badge>}
          {data.decision && <Badge tone="good">Human decision saved</Badge>}
        </div>
        <code className="request-id">{data.request_id}</code>
      </div>

      <Card title="Raw transcript">
        <div className="transcript">{data.transcript}</div>
      </Card>

      <div className="comparison-grid">
        <Card title="Production output">
          <p className="model-line">{data.production_model || "Unknown production model"}</p>
          <DiffText base={data.transcript} target={data.production_output} label="Compared with raw" />
        </Card>
        <Card title="Luna proposal">
          {proposal ? (
            <>
              <p className="model-line">{proposal.model || "gpt-5.6-luna"}</p>
              <DiffText base={data.production_output} target={proposal.text} label="Compared with production" />
            </>
          ) : (
            <div className="analysis-pending">
              {data.job?.state === "failed" ? (
                <>
                  <Badge tone="bad">Analysis failed</Badge>
                  <p>{data.job.error}</p>
                  <button className="button secondary" onClick={() => retryAnalysis.mutate()}>
                    Retry Luna
                  </button>
                </>
              ) : (
                <Loading label="Waiting for Luna" />
              )}
            </div>
          )}
        </Card>
      </div>

      {analysis && (
        <Card title="Independent Luna evaluation">
          <div className="evaluation-summary">
            <div className="badge-stack horizontal">
              <Badge tone={analysis.validation_status === "pass" ? "good" : "bad"}>
                Validation {analysis.validation_status}
              </Badge>
              <Badge tone="info">Preference · {analysis.preference}</Badge>
            </div>
            <p className="evaluation-reason">{analysis.pairwise_reason}</p>
          </div>
          <ScoreGrid
            production={analysis.production_scores}
            proposal={analysis.proposal_scores}
          />
          {(analysis.validation_reason || analysis.validation_type) && (
            <p className="muted">
              {analysis.validation_type} {analysis.validation_reason}
            </p>
          )}
        </Card>
      )}

      <Card title="Final label">
        <textarea
          className="label-editor"
          aria-label="Final training label"
          value={editor}
          onChange={(event) => {
            setEditor(event.target.value);
            setEditorDirty(true);
          }}
        />
        <div className="decision-controls">
          <label>
            Context dependency
            <select
              value={contextDependency}
              onChange={(event) => setContextDependency(event.target.value)}
            >
              <option value="unknown">Unknown</option>
              <option value="none">None</option>
              <option value="window">Window context</option>
              <option value="clipboard">Clipboard context</option>
              <option value="vocabulary">Vocabulary</option>
            </select>
          </label>
          <div className="button-row decision-buttons">
            <button
              className="button primary"
              disabled={!proposal || decision.isPending}
              onClick={() => decision.mutate("accept_luna")}
            >
              Accept Luna
            </button>
            <button
              className="button secondary"
              disabled={!editor.trim() || decision.isPending}
              onClick={() => decision.mutate("human_edit")}
            >
              Save my edit
            </button>
            <button
              className="button secondary"
              disabled={decision.isPending}
              onClick={() => decision.mutate("production_correct")}
            >
              Production was correct
            </button>
            <button
              className="button danger"
              disabled={decision.isPending}
              onClick={() => decision.mutate("exclude")}
            >
              Exclude
            </button>
          </div>
        </div>
        {decision.isPending && <p className="save-status">Saving durable decision…</p>}
        {decision.error && <ErrorState error={decision.error} />}
      </Card>

      <details className="context-panel">
        <summary>Session context</summary>
        <div className="context-grid">
          <div><strong>Window</strong><pre>{data.window_context || "None"}</pre></div>
          <div><strong>Clipboard</strong><pre>{data.clipboard_context || "None"}</pre></div>
          <div><strong>Vocabulary</strong><pre>{data.custom_vocabulary || "None"}</pre></div>
        </div>
      </details>
    </Page>
  );
}
