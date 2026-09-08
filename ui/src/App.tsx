import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { OverviewPage } from "./pages/OverviewPage";
import { QueuePage } from "./pages/QueuePage";
import { SampleListPage } from "./pages/SampleListPage";
import { SamplePage } from "./pages/SamplePage";
import {
  CohortPage,
  CohortsPage,
  ReleasePage,
  ReleasesPage,
} from "./pages/DataPages";
import {
  DocsPage,
  ModelPage,
  ModelsPage,
  RunPage,
  RunsPage,
  SystemPage,
} from "./pages/OperationsPages";

export function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/review/recent" element={<SampleListPage scope="recent" />} />
        <Route path="/review/queue" element={<QueuePage />} />
        <Route path="/review/history" element={<SampleListPage scope="history" />} />
        <Route path="/samples/:requestId" element={<SamplePage />} />
        <Route path="/data/cohorts" element={<CohortsPage />} />
        <Route path="/data/cohorts/:name" element={<CohortPage />} />
        <Route path="/data/releases" element={<ReleasesPage />} />
        <Route path="/data/releases/:name" element={<ReleasePage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/models/:name" element={<ModelPage />} />
        <Route path="/runs/training" element={<RunsPage category="training" />} />
        <Route path="/runs/evaluations" element={<RunsPage category="evaluations" />} />
        <Route path="/runs/:runId" element={<RunPage />} />
        <Route path="/system" element={<SystemPage />} />
        <Route path="/system/docs" element={<DocsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
