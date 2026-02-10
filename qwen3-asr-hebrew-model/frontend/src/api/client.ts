import type { Dataset, Model, ResultFile, ResultData, JobStatus, EvaluationRequest } from './types'

// Use same hostname as frontend, different port
const API_BASE = `http://${window.location.hostname}:8001/api`

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || response.statusText)
  }
  return response.json()
}

export const api = {
  // Health check
  health: () => fetchJson<{ status: string }>(`${API_BASE}/health`),

  // Datasets
  getDatasets: () => fetchJson<Dataset[]>(`${API_BASE}/datasets`),

  // Models
  getModels: () => fetchJson<Model[]>(`${API_BASE}/models`),

  // Results
  getResults: () => fetchJson<ResultFile[]>(`${API_BASE}/results`),
  getResult: (filename: string) => fetchJson<ResultData>(`${API_BASE}/results/${filename}`),

  // Evaluation
  startEvaluation: (request: EvaluationRequest) =>
    fetchJson<JobStatus>(`${API_BASE}/evaluate`, {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  getJobStatus: (jobId: string) => fetchJson<JobStatus>(`${API_BASE}/evaluate/${jobId}/status`),

  cancelJob: (jobId: string) =>
    fetchJson<{ status: string; job_id: string }>(`${API_BASE}/evaluate/${jobId}/cancel`, {
      method: 'POST',
    }),

  // SSE stream URL
  getStreamUrl: (jobId: string) => `${API_BASE}/evaluate/stream/${jobId}`,

  // Audio URL
  getAudioUrl: (dataset: string, sampleId: number) => `${API_BASE}/audio/${dataset}/${sampleId}`,

  // Equivalences
  getEquivalences: () => fetchJson<{ rules: { canonical: string; words: string[] }[] }>(`${API_BASE}/equivalences`),

  addEquivalence: (word1: string, word2: string) =>
    fetchJson<{ status: string; canonical: string; words: string[] }>(`${API_BASE}/equivalences`, {
      method: 'POST',
      body: JSON.stringify({ word1, word2 }),
    }),

  removeEquivalence: (word: string) =>
    fetchJson<{ status: string }>(`${API_BASE}/equivalences/${encodeURIComponent(word)}`, {
      method: 'DELETE',
    }),

  // Checkpoints
  getCheckpoints: () =>
    fetchJson<{ checkpoints: { dataset: string; model: string; completed: number; total: number; timestamp: string }[] }>(
      `${API_BASE}/checkpoints`
    ),

  deleteCheckpoint: (dataset: string, model: string) =>
    fetchJson<{ status: string }>(`${API_BASE}/checkpoints/${dataset}/${encodeURIComponent(model)}`, {
      method: 'DELETE',
    }),

  // Recalculate WER with current equivalences and optional temporary ignored words
  recalculateWer: (entries: unknown[], tempIgnoredWords?: string[]) =>
    fetchJson<{ entries: unknown[]; avg_wer: number }>(`${API_BASE}/recalculate-wer`, {
      method: 'POST',
      body: JSON.stringify({ entries, temp_ignored_words: tempIgnoredWords }),
    }),

  // Ignored words
  getIgnoredWords: () => fetchJson<{ words: string[] }>(`${API_BASE}/ignored-words`),

  addIgnoredWord: (word: string) =>
    fetchJson<{ status: string; word: string }>(`${API_BASE}/ignored-words/${encodeURIComponent(word)}`, {
      method: 'POST',
    }),

  removeIgnoredWord: (word: string) =>
    fetchJson<{ status: string }>(`${API_BASE}/ignored-words/${encodeURIComponent(word)}`, {
      method: 'DELETE',
    }),
}
