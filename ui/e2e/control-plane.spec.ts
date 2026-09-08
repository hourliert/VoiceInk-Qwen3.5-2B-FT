import { expect, test } from "@playwright/test";

const stats = {
  samples: 11131,
  annotations: 6981,
  analyses: 1346,
  reviewed: 4419,
  training_eligible: 4419,
  pending_jobs: 0,
  failed_jobs: 0,
  outcomes: {},
};

function sample(id: string, transcript: string) {
  return {
    request_id: id,
    timestamp: "2026-08-31T10:00:00Z",
    transcript,
    custom_vocabulary: "",
    clipboard_context: "",
    window_context: "",
    production_model: "Qwen",
    production_output: "Production output.",
    annotations: [{
      id: id === "one" ? 7 : 17,
      text: "Luna proposal.",
      origin: "luna",
      model: "gpt-5.6-luna",
      reasoning_effort: "xhigh",
      created_at: "2026-08-31T10:01:00Z",
    }],
    analyses: [{
      id: id === "one" ? 8 : 18,
      annotation_id: id === "one" ? 7 : 17,
      validation_status: "pass",
      validation_type: "",
      validation_reason: "",
      production_scores: {meaning_preservation: 4},
      proposal_scores: {meaning_preservation: 5},
      preference: "proposal",
      confidence: "high",
      material_difference: true,
      context_analysis: {},
      score_analysis: {},
      pairwise_reason: "Proposal is more faithful.",
      created_at: "2026-08-31T10:02:00Z",
    }],
    decision: null,
    job: {id: 9, state: "completed", attempts: 1, error: "", updated_at: ""},
  };
}

test("a delayed old route never replaces the current page", async ({page}) => {
  await page.route("**/api/v1/overview", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 700));
    await route.fulfill({json: {
      stats,
      active_cohort: null,
      latest_release: null,
      production_model: "late-model",
    }});
  });
  await page.route("**/api/v1/samples**", (route) => route.fulfill({json: {items: []}}));

  await page.goto("/");
  await page.getByRole("link", {name: "Recent"}).click();
  await expect(page).toHaveURL(/\/review\/recent/);
  await expect(page.getByRole("heading", {name: "Recent VoiceInk use"})).toBeVisible();
  await page.waitForTimeout(900);
  await expect(page.getByRole("heading", {name: "Recent VoiceInk use"})).toBeVisible();
  await expect(page.getByText("late-model")).toHaveCount(0);
});

test("saving advances to the next Luna-ready required sample", async ({page}) => {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path === "/api/v1/samples/one" && route.request().method() === "GET") {
      await route.fulfill({json: sample("one", "First raw transcript.")});
      return;
    }
    if (path === "/api/v1/samples/two" && route.request().method() === "GET") {
      await route.fulfill({json: sample("two", "Second raw transcript.")});
      return;
    }
    if (path.endsWith("/decisions")) {
      await route.fulfill({status: 201, json: {
        ok: true, decision_id: 1, annotation_id: 7,
        maintenance: "scheduled", maintenance_job_id: 1,
      }});
      return;
    }
    if (path.endsWith("/queue")) {
      await route.fulfill({json: {
        cohort: "voiceink-bootstrap-v1", total: 1, offset: 0, limit: 500,
        items: [{
          request_id: "two", timestamp: "2026-08-31T10:00:00Z",
          transcript: "Second raw transcript.", production_model: "Qwen",
          production_output: "Production output.", split: "train",
          stratum: "representative", audit_selected: 0, fresh_luna: true,
          human_reviewed: false, requires_human: true, review_pending: true,
        }],
      }});
      return;
    }
    if (path === "/api/v1/active-cohort") {
      await route.fulfill({json: {name: "voiceink-bootstrap-v1"}});
      return;
    }
    await route.fulfill({json: {}});
  });

  await page.goto("/samples/one?cohort=voiceink-bootstrap-v1&review=required");
  await expect(page.getByText("First raw transcript.")).toBeVisible();
  await page.getByRole("button", {name: "Accept Luna"}).click();
  await expect(page).toHaveURL(/\/samples\/two/);
  await expect(page.getByText("Second raw transcript.")).toBeVisible();
});


test("a pending Luna job updates in place and preserves the editor", async ({page}) => {
  let sampleReads = 0;
  let analysisReady = false;
  const pending = {
    ...sample("polling", "A transcript waiting for Luna."),
    annotations: [],
    analyses: [],
    job: {id: 19, state: "pending", attempts: 0, error: "", updated_at: ""},
  };
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/samples/polling") {
      sampleReads += 1;
      await route.fulfill({
        json: analysisReady
          ? sample("polling", "A transcript waiting for Luna.")
          : pending,
      });
      return;
    }
    if (url.pathname === "/api/v1/active-cohort") {
      await route.fulfill({json: {name: "voiceink-bootstrap-v1"}});
      return;
    }
    await route.fulfill({json: {}});
  });

  await page.goto("/samples/polling?cohort=voiceink-bootstrap-v1");
  await expect(page.getByText("Luna running", {exact: true})).toBeVisible();
  const editor = page.getByLabel("Final training label");
  await editor.fill("My in-progress correction.");
  analysisReady = true;

  await expect(page.getByText("Luna ready", {exact: true})).toBeVisible({timeout: 5_000});
  await expect(page.getByText("Independent Luna evaluation", {exact: true})).toBeVisible();
  await expect(editor).toHaveValue("My in-progress correction.");
  expect(sampleReads).toBeGreaterThanOrEqual(2);
});
