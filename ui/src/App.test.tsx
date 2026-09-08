import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import type { Sample } from "./types";

function response(value: unknown) {
  return Promise.resolve(new Response(JSON.stringify(value), {
    status: 200,
    headers: {"Content-Type": "application/json"},
  }));
}

function renderApp(path: string, client = new QueryClient({
  defaultOptions: {queries: {retry: false, staleTime: 0}},
})) {
  return {
    client,
    ...render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("route-scoped rendering", () => {
  it("does not let a delayed overview overwrite a newer route", async () => {
    let resolveOverview: (value: Response) => void = () => undefined;
    const delayedOverview = new Promise<Response>((resolve) => {
      resolveOverview = resolve;
    });
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/overview")) return delayedOverview;
      if (url.includes("/samples")) return response({items: []});
      throw new Error("Unexpected request: " + url);
    }));

    renderApp("/");
    fireEvent.click(screen.getByRole("link", {name: "Recent"}));
    await screen.findByRole("heading", {name: "Recent VoiceInk use"});
    resolveOverview(new Response(JSON.stringify({
      stats: {samples: 1, annotations: 1, analyses: 0, reviewed: 0,
        training_eligible: 0, pending_jobs: 0, failed_jobs: 0, outcomes: {}},
      active_cohort: null,
      latest_release: null,
      production_model: "late-model",
    }), {status: 200, headers: {"Content-Type": "application/json"}}));

    await act(async () => { await delayedOverview; });
    expect(screen.getByRole("heading", {name: "Recent VoiceInk use"})).toBeTruthy();
    expect(screen.queryByText("late-model")).toBeNull();
  });

  it("preserves a dirty label editor across server-state updates", async () => {
    const sample: Sample = {
      request_id: "sample-1",
      timestamp: "2026-08-31T10:00:00Z",
      transcript: "Raw transcript.",
      custom_vocabulary: "",
      clipboard_context: "",
      window_context: "",
      production_model: "Qwen",
      production_output: "Production output.",
      annotations: [{
        id: 7, text: "Luna proposal.", origin: "luna", model: "Luna",
        reasoning_effort: "xhigh", created_at: "2026-08-31T10:01:00Z",
      }],
      analyses: [{
        id: 8, annotation_id: 7, validation_status: "pass",
        validation_type: "", validation_reason: "",
        production_scores: {meaning_preservation: 4},
        proposal_scores: {meaning_preservation: 5},
        preference: "proposal", confidence: "high",
        material_difference: true, context_analysis: {}, score_analysis: {},
        pairwise_reason: "Proposal is better.", created_at: "2026-08-31T10:02:00Z",
      }],
      decision: null,
      job: {id: 9, state: "completed", attempts: 1, error: "", updated_at: ""},
    };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/samples/sample-1")) return response(sample);
      if (url.includes("/active-cohort")) return response({name: "cohort"});
      throw new Error("Unexpected request: " + url);
    }));
    const {client} = renderApp("/samples/sample-1");

    const editor = await screen.findByRole("textbox", {name: "Final training label"});
    fireEvent.change(editor, {target: {value: "My careful correction."}});
    act(() => {
      client.setQueryData(["sample", "sample-1"], {
        ...sample,
        production_output: "A background update.",
      });
    });

    await waitFor(() => {
      expect((screen.getByRole("textbox", {
        name: "Final training label",
      }) as HTMLTextAreaElement).value).toBe("My careful correction.");
    });
  });
});
