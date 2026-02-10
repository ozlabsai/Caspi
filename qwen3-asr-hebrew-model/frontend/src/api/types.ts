export interface Dataset {
  id: string
  name: string
  split: string
  text_col: string
  description: string
}

export interface Model {
  id: string
  name: string
  default: boolean
}

export interface EvaluationEntry {
  id: number
  dataset?: string
  dataset_sample_idx?: number  // Index within dataset for audio playback
  reference_text: string
  predicted_text: string
  norm_reference_text?: string
  norm_predicted_text?: string
  wer: number
  wil: number
  audio_duration: number
  transcription_time: number
  substitutions?: number
  deletions?: number
  insertions?: number
  hits?: number
}

export interface JobStatus {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'cancelled' | 'error'
  dataset: string
  model: string
  total_samples: number
  completed_samples: number
  current_wer: number
  start_time: string | null
  end_time: string | null
  output_file: string | null
  error: string | null
}

export interface ResultFile {
  filename: string
  path: string
  size: number
  modified: string
  dataset: string | null
  model: string | null
  wer: number | null
  samples: number | null
}

export interface ResultData {
  rows: EvaluationEntry[]
  summary: {
    total_samples: number
    avg_wer: number | null
    total_audio_duration: number | null
    total_transcription_time: number | null
    rtf: number | null
  }
}

export interface SSEProgressEvent {
  event: 'connected' | 'progress' | 'complete' | 'error' | 'cancelled' | 'keepalive' | 'started'
  job_id?: string
  completed?: number
  total?: number
  current_wer?: number
  current_dataset?: string
  entry?: EvaluationEntry
  final_wer?: number
  output_file?: string
  error?: string
}

export interface EvaluationRequest {
  dataset: string
  model: string
  max_samples?: number
  device?: string
  resume?: boolean
}

export interface Checkpoint {
  dataset: string
  model: string
  completed: number
  total: number
  timestamp: string
}
