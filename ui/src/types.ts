export type JsonObject = Record<string, unknown>;

export interface Stats {
  samples: number;
  annotations: number;
  analyses: number;
  reviewed: number;
  training_eligible: number;
  pending_jobs: number;
  failed_jobs: number;
  outcomes: Record<string, number>;
}

export interface CohortStatus {
  name: string;
  state: string;
  ready: boolean;
  members: number;
  fresh_luna: number;
  human_reviewed: number;
  required_human: number;
  required_human_done: number;
  unresolved_critical: number;
  expanded_review_strata: string[];
  splits: Record<string, number>;
}

export interface Cohort {
  id: number;
  name: string;
  state: string;
  created_at: string;
  updated_at?: string;
  status: CohortStatus;
}

export interface ReleasePreview {
  cohort: string;
  correction_cutoff: string;
  additions: Array<{request_id: string; annotation_id: number; decided_at: string}>;
  splits: Record<string, number>;
  records: number;
  selection_sha256: string;
}

export interface SampleListItem {
  request_id: string;
  timestamp: string;
  transcript: string;
  production_model: string;
  production_output: string;
  duration_ms?: number;
  outcome?: string;
  training_eligible?: boolean;
  job_state?: string;
}

export interface QueueItem {
  request_id: string;
  timestamp: string;
  transcript: string;
  production_model: string;
  production_output: string;
  split: string;
  stratum: string;
  audit_selected: number;
  fresh_luna: boolean;
  human_reviewed: boolean;
  requires_human: boolean;
  review_pending: boolean;
  job_state?: string;
  job_error?: string;
  validation_status?: string;
  preference?: string;
  confidence?: string;
  material_difference?: boolean;
}

export interface QueueResponse {
  cohort: string;
  total: number;
  offset: number;
  limit?: number;
  items: QueueItem[];
}

export interface Annotation {
  id: number;
  text: string;
  origin: string;
  model: string;
  reasoning_effort: string;
  created_at: string;
  metadata?: JsonObject;
}

export interface Analysis {
  id: number;
  annotation_id: number;
  validation_status: string;
  validation_type: string;
  validation_reason: string;
  production_scores: Record<string, number>;
  proposal_scores: Record<string, number>;
  preference: string;
  confidence: string;
  material_difference: boolean;
  context_analysis: Record<string, string>;
  score_analysis: Record<string, string>;
  pairwise_reason: string;
  created_at: string;
}

export interface Decision {
  id: number;
  outcome: string;
  annotation_id: number | null;
  reviewer: string;
  context_dependency: string;
  decided_at: string;
}

export interface AnalysisJob {
  id: number;
  state: string;
  attempts: number;
  error: string;
  updated_at: string;
}

export interface Sample {
  request_id: string;
  timestamp: string;
  transcript: string;
  custom_vocabulary: string;
  clipboard_context: string;
  window_context: string;
  production_model: string;
  production_output: string;
  duration_ms?: number;
  annotations: Annotation[];
  analyses: Analysis[];
  decision: Decision | null;
  job: AnalysisJob | null;
}

export interface RunSummary {
  run_id: string;
  experiment_id: string;
  experiment: string;
  name: string;
  status: string;
  start_time: string;
  end_time: string;
  kind: string;
  release: string;
  profile: string;
  baseline: string;
  candidate: string;
  metrics: Record<string, number>;
  mlflow_url: string;
}

export interface Overview {
  stats: Stats;
  active_cohort: Cohort | null;
  latest_release: JsonObject | null;
  production_model: string;
}
