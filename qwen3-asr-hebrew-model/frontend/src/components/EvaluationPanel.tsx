import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { Play, Square, Loader2, Volume2, ChevronUp, ChevronDown, Pause } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { api } from '@/api/client'
import { useSSE } from '@/hooks/useSSE'
import type { Dataset, Model, JobStatus, EvaluationEntry, SSEProgressEvent, Checkpoint } from '@/api/types'

// Word-level diff for highlighting differences
type DiffType = 'equal' | 'insert' | 'delete' | 'replace'
interface DiffWord {
  word: string
  type: DiffType
}

function computeWordDiff(reference: string, predicted: string): { ref: DiffWord[], pred: DiffWord[] } {
  const refWords = reference.split(/\s+/).filter(Boolean)
  const predWords = predicted.split(/\s+/).filter(Boolean)

  // Simple LCS-based diff
  const m = refWords.length
  const n = predWords.length

  // Build LCS table
  const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0))
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (refWords[i - 1] === predWords[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  // Backtrack to find diff
  const refResult: DiffWord[] = []
  const predResult: DiffWord[] = []
  let i = m, j = n

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && refWords[i - 1] === predWords[j - 1]) {
      refResult.unshift({ word: refWords[i - 1], type: 'equal' })
      predResult.unshift({ word: predWords[j - 1], type: 'equal' })
      i--
      j--
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      predResult.unshift({ word: predWords[j - 1], type: 'insert' })
      j--
    } else {
      refResult.unshift({ word: refWords[i - 1], type: 'delete' })
      i--
    }
  }

  return { ref: refResult, pred: predResult }
}

interface DiffViewProps {
  reference: string
  predicted: string
  onMarkEquivalent?: (refWord: string, predWord: string) => void
  onIgnoreWord?: (word: string) => void  // Local ignore (current sample only)
  onIgnoreWordGlobal?: (word: string) => void  // Global ignore (saves & propagates)
}

function DiffView({ reference, predicted, onMarkEquivalent, onIgnoreWord, onIgnoreWordGlobal }: DiffViewProps) {
  const diff = useMemo(() => computeWordDiff(reference, predicted), [reference, predicted])
  const [selectedRef, setSelectedRef] = useState<string | null>(null)
  const [selectedPred, setSelectedPred] = useState<string | null>(null)

  const handleRefClick = (e: React.MouseEvent, word: string, type: DiffType) => {
    if (type !== 'delete') return

    // Shift+click to ignore globally (saves & propagates)
    if (e.shiftKey && onIgnoreWordGlobal) {
      onIgnoreWordGlobal(word)
      return
    }

    // Option/Alt+click to ignore locally (current sample only)
    if (e.altKey && onIgnoreWord) {
      onIgnoreWord(word)
      return
    }

    if (!onMarkEquivalent) return
    if (selectedPred) {
      onMarkEquivalent(word, selectedPred)
      setSelectedRef(null)
      setSelectedPred(null)
    } else {
      setSelectedRef(word)
    }
  }

  const handlePredClick = (e: React.MouseEvent, word: string, type: DiffType) => {
    if (type !== 'insert') return

    // Shift+click to ignore globally (saves & propagates)
    if (e.shiftKey && onIgnoreWordGlobal) {
      onIgnoreWordGlobal(word)
      return
    }

    // Option/Alt+click to ignore locally (current sample only)
    if (e.altKey && onIgnoreWord) {
      onIgnoreWord(word)
      return
    }

    if (!onMarkEquivalent) return
    if (selectedRef) {
      onMarkEquivalent(selectedRef, word)
      setSelectedRef(null)
      setSelectedPred(null)
    } else {
      setSelectedPred(word)
    }
  }

  return (
    <div className="space-y-2">
      {onMarkEquivalent && (selectedRef || selectedPred) && (
        <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
          Selected: <strong>{selectedRef || selectedPred}</strong> — click a {selectedRef ? 'green' : 'red'} word to mark as equivalent
          <button className="ml-2 text-gray-500 hover:text-gray-700" onClick={() => { setSelectedRef(null); setSelectedPred(null) }}>
            (cancel)
          </button>
        </div>
      )}
      {(onMarkEquivalent || onIgnoreWord || onIgnoreWordGlobal) && (
        <div className="text-xs text-muted-foreground">
          Click to select for equivalence • <strong>Option/Alt+click</strong> to ignore (this sample) • <strong>Shift+click</strong> to ignore (global)
        </div>
      )}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs font-medium text-muted-foreground mb-1 block">Reference</label>
          <div className="p-3 bg-muted/50 rounded text-sm leading-relaxed" dir="rtl">
            {diff.ref.map((w, i) => (
              <span
                key={i}
                onClick={(e) => handleRefClick(e, w.word, w.type)}
                className={`${
                  w.type === 'delete'
                    ? `bg-red-200 text-red-900 px-0.5 rounded ${onMarkEquivalent || onIgnoreWord ? 'cursor-pointer hover:bg-red-300' : ''}`
                    : ''
                } ${selectedRef === w.word ? 'ring-2 ring-blue-500' : ''}`}
              >
                {w.word}{' '}
              </span>
            ))}
          </div>
        </div>
        <div>
          <label className="text-xs font-medium text-muted-foreground mb-1 block">Predicted</label>
          <div className="p-3 bg-muted/50 rounded text-sm leading-relaxed" dir="rtl">
            {diff.pred.map((w, i) => (
              <span
                key={i}
                onClick={(e) => handlePredClick(e, w.word, w.type)}
                className={`${
                  w.type === 'insert'
                    ? `bg-green-200 text-green-900 px-0.5 rounded ${onMarkEquivalent || onIgnoreWord ? 'cursor-pointer hover:bg-green-300' : ''}`
                    : ''
                } ${selectedPred === w.word ? 'ring-2 ring-blue-500' : ''}`}
              >
                {w.word}{' '}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export function EvaluationPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [selectedDataset, setSelectedDataset] = useState('all')
  const [selectedModel, setSelectedModel] = useState('')
  const [maxSamples, setMaxSamples] = useState<number | ''>('')
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [entries, setEntries] = useState<EvaluationEntry[]>([])
  const [currentEntry, setCurrentEntry] = useState<EvaluationEntry | null>(null)
  const [currentDataset, setCurrentDataset] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedEntry, setSelectedEntry] = useState<EvaluationEntry | null>(null)
  const [checkpoint, setCheckpoint] = useState<Checkpoint | null>(null)
  const [sortBy, setSortBy] = useState<'id' | 'wer' | 'duration' | 'time' | 'rtf'>('id')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [filterDataset, setFilterDataset] = useState<string>('all')
  const [isPlaying, setIsPlaying] = useState(false)
  const [localIgnoredWords, setLocalIgnoredWords] = useState<Record<string, string[]>>(() => {
    // Load from localStorage on init
    try {
      const stored = localStorage.getItem('localIgnoredWords')
      return stored ? JSON.parse(stored) : {}
    } catch {
      return {}
    }
  })
  const tableRef = useRef<HTMLDivElement>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // Save to localStorage whenever localIgnoredWords changes
  useEffect(() => {
    try {
      localStorage.setItem('localIgnoredWords', JSON.stringify(localIgnoredWords))
    } catch {
      // Ignore storage errors
    }
  }, [localIgnoredWords])

  // Load datasets and models
  useEffect(() => {
    Promise.all([api.getDatasets(), api.getModels()])
      .then(([ds, ms]) => {
        setDatasets(ds)
        setModels(ms)
        setSelectedModel(ms.find(m => m.default)?.id || ms[0]?.id || '')
      })
      .catch(e => setError(e.message))
  }, [])

  // Check for available checkpoint when dataset/model changes
  useEffect(() => {
    if (selectedDataset && selectedModel) {
      api.getCheckpoints().then(({ checkpoints }) => {
        const cp = checkpoints.find(c => c.dataset === selectedDataset && c.model === selectedModel)
        setCheckpoint(cp || null)
      }).catch(() => setCheckpoint(null))
    }
  }, [selectedDataset, selectedModel])

  // Handler for marking words as equivalent
  const handleMarkEquivalent = useCallback(async (refWord: string, predWord: string) => {
    try {
      await api.addEquivalence(refWord, predWord)

      // Recalculate WER for all current entries
      if (entries.length > 0) {
        const result = await api.recalculateWer(entries)
        setEntries(result.entries as EvaluationEntry[])

        // Update job status with new average WER
        setJobStatus(prev => prev ? {
          ...prev,
          current_wer: result.avg_wer,
        } : null)

        // Update selected entry if it was affected
        if (selectedEntry) {
          const updated = (result.entries as EvaluationEntry[]).find(e => e.id === selectedEntry.id)
          if (updated) {
            setSelectedEntry(updated)
          }
        }
      }

      alert(`Marked "${refWord}" and "${predWord}" as equivalent. WER recalculated for all samples.`)
    } catch (e) {
      setError(`Failed to add equivalence: ${(e as Error).message}`)
    }
  }, [entries, selectedEntry])

  // Create a stable key for an entry (dataset + reference text hash for uniqueness)
  const getEntryKey = useCallback((entry: EvaluationEntry): string => {
    return `${entry.dataset || 'unknown'}_${entry.id}_${entry.reference_text.slice(0, 50)}`
  }, [])

  // Handler for ignoring words (only affects current sample, does NOT propagate)
  const handleIgnoreWord = useCallback(async (word: string) => {
    if (!selectedEntry) return

    try {
      // Create a unique key for this entry
      const entryKey = getEntryKey(selectedEntry)
      const existingWords = localIgnoredWords[entryKey] || []
      const updatedWords = [...new Set([...existingWords, word])]

      // Update state (this will also save to localStorage via useEffect)
      setLocalIgnoredWords(prev => ({
        ...prev,
        [entryKey]: updatedWords
      }))

      // Recalculate WER with ALL accumulated ignored words for this entry
      const result = await api.recalculateWer([selectedEntry], updatedWords)
      const updatedEntry = (result.entries as EvaluationEntry[])[0]

      if (updatedEntry) {
        // Update only the selected entry in the entries list
        setEntries(prev => prev.map(e => e.id === updatedEntry.id ? updatedEntry : e))
        setSelectedEntry(updatedEntry)

        // Recalculate average WER from all entries
        setJobStatus(prev => {
          if (!prev) return null
          const allEntries = entries.map(e => e.id === updatedEntry.id ? updatedEntry : e)
          const avgWer = allEntries.reduce((sum, e) => sum + e.wer, 0) / allEntries.length
          return { ...prev, current_wer: avgWer }
        })
      }

      alert(`Word "${word}" ignored for this sample (${updatedWords.length} total). WER: ${(updatedEntry.wer * 100).toFixed(1)}%`)
    } catch (e) {
      setError(`Failed to ignore word: ${(e as Error).message}`)
    }
  }, [entries, selectedEntry, localIgnoredWords, getEntryKey])

  // Handler for ignoring words globally (saves to file, propagates to ALL samples)
  const handleIgnoreWordGlobal = useCallback(async (word: string) => {
    try {
      await api.addIgnoredWord(word)

      // Recalculate WER for all current entries
      if (entries.length > 0) {
        const result = await api.recalculateWer(entries)
        setEntries(result.entries as EvaluationEntry[])

        // Update job status with new average WER
        setJobStatus(prev => prev ? {
          ...prev,
          current_wer: result.avg_wer,
        } : null)

        // Update selected entry if it was affected
        if (selectedEntry) {
          const updated = (result.entries as EvaluationEntry[]).find(e => e.id === selectedEntry.id)
          if (updated) {
            setSelectedEntry(updated)
          }
        }
      }

      alert(`Word "${word}" added to global ignore list. WER recalculated for all samples.`)
    } catch (e) {
      setError(`Failed to ignore word globally: ${(e as Error).message}`)
    }
  }, [entries, selectedEntry])

  const handleProgress = useCallback((event: SSEProgressEvent) => {
    if (event.entry) {
      setCurrentEntry(event.entry)
      setEntries(prev => [...prev, event.entry!])
      // Auto-scroll to bottom
      setTimeout(() => {
        if (tableRef.current) {
          tableRef.current.scrollTop = tableRef.current.scrollHeight
        }
      }, 100)
    }
    if (event.current_dataset) {
      setCurrentDataset(event.current_dataset)
    }
    setJobStatus(prev => prev ? {
      ...prev,
      completed_samples: event.completed || prev.completed_samples,
      total_samples: event.total || prev.total_samples,
      current_wer: event.current_wer ?? prev.current_wer,
      status: 'running',
    } : null)
  }, [])

  const handleComplete = useCallback((event: SSEProgressEvent) => {
    setJobStatus(prev => prev ? {
      ...prev,
      status: 'completed',
      current_wer: event.final_wer ?? prev.current_wer,
      output_file: event.output_file || prev.output_file,
    } : null)
    setCurrentEntry(null)
    setJobId(null)
  }, [])

  const handleError = useCallback((errorMsg: string) => {
    setError(errorMsg)
    setJobStatus(prev => prev ? { ...prev, status: 'error', error: errorMsg } : null)
    setJobId(null)
  }, [])

  // Debug: log when jobId changes
  useEffect(() => {
    console.log('jobId changed to:', jobId)
  }, [jobId])

  useSSE(jobId, { onProgress: handleProgress, onComplete: handleComplete, onError: handleError })

  const startEvaluation = async (resume = false) => {
    console.log('Starting evaluation...', resume ? '(resuming)' : '')
    setLoading(true)
    setError(null)
    if (!resume) {
      setEntries([])
    }
    setCurrentEntry(null)
    setCurrentDataset('')
    try {
      const status = await api.startEvaluation({
        dataset: selectedDataset,
        model: selectedModel,
        max_samples: maxSamples || undefined,
        resume,
      })
      console.log('Got job status:', status)
      console.log('Setting jobId to:', status.job_id)
      setJobStatus(status)
      setJobId(status.job_id)
      if (resume) {
        setCheckpoint(null) // Clear checkpoint indicator after resuming
      }
    } catch (e) {
      console.error('Start evaluation error:', e)
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const cancelEvaluation = async () => {
    if (!jobId) return
    try {
      await api.cancelJob(jobId)
      setJobStatus(prev => prev ? { ...prev, status: 'cancelled' } : null)
      setJobId(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const isRunning = jobStatus?.status === 'running' || jobStatus?.status === 'pending'
  const progress = jobStatus && jobStatus.total_samples > 0
    ? (jobStatus.completed_samples / jobStatus.total_samples) * 100
    : 0

  const getWerColor = (wer: number) => {
    if (wer < 0.1) return 'text-green-600 bg-green-50'
    if (wer < 0.2) return 'text-lime-600 bg-lime-50'
    if (wer < 0.3) return 'text-yellow-600 bg-yellow-50'
    if (wer < 0.5) return 'text-orange-600 bg-orange-50'
    return 'text-red-600 bg-red-50'
  }

  // Filtered and sorted entries
  const filteredEntries = useMemo(() => {
    if (filterDataset === 'all') return entries
    return entries.filter(e => e.dataset === filterDataset)
  }, [entries, filterDataset])

  const sortedEntries = useMemo(() => {
    const sorted = [...filteredEntries]
    sorted.sort((a, b) => {
      let aVal: number, bVal: number
      switch (sortBy) {
        case 'wer':
          aVal = a.wer
          bVal = b.wer
          break
        case 'duration':
          aVal = a.audio_duration ?? 0
          bVal = b.audio_duration ?? 0
          break
        case 'time':
          aVal = a.transcription_time ?? 0
          bVal = b.transcription_time ?? 0
          break
        case 'rtf':
          aVal = (a.transcription_time ?? 0) / (a.audio_duration || 1)
          bVal = (b.transcription_time ?? 0) / (b.audio_duration || 1)
          break
        default:
          aVal = a.id
          bVal = b.id
      }
      return sortDir === 'asc' ? aVal - bVal : bVal - aVal
    })
    return sorted
  }, [filteredEntries, sortBy, sortDir])

  // Get unique datasets for filter dropdown
  const uniqueDatasets = useMemo(() => {
    const datasets = new Set(entries.map(e => e.dataset).filter(Boolean))
    return Array.from(datasets).sort()
  }, [entries])

  const toggleSort = (col: typeof sortBy) => {
    if (sortBy === col) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(col)
      setSortDir('asc')
    }
  }

  const SortIcon = ({ col }: { col: typeof sortBy }) => {
    if (sortBy !== col) return null
    return sortDir === 'asc' ? <ChevronUp className="h-3 w-3 inline ml-1" /> : <ChevronDown className="h-3 w-3 inline ml-1" />
  }

  // Calculate WER per dataset
  const werByDataset = useMemo(() => {
    const grouped: Record<string, { total: number; werSum: number; count: number }> = {}
    for (const e of entries) {
      const ds = e.dataset || 'unknown'
      if (!grouped[ds]) {
        grouped[ds] = { total: 0, werSum: 0, count: 0 }
      }
      grouped[ds].werSum += e.wer
      grouped[ds].count += 1
    }
    return Object.entries(grouped).map(([dataset, stats]) => ({
      dataset,
      avgWer: stats.count > 0 ? stats.werSum / stats.count : 0,
      count: stats.count,
    })).sort((a, b) => a.dataset.localeCompare(b.dataset))
  }, [entries])

  // Calculate timing benchmarks like ivrit.ai format
  const timingBenchmarks = useMemo(() => {
    const entriesToUse = filterDataset === 'all' ? entries : filteredEntries

    // Helper to calculate percentiles
    const percentile = (arr: number[], p: number) => {
      if (arr.length === 0) return 0
      const sorted = [...arr].sort((a, b) => a - b)
      const idx = Math.ceil((p / 100) * sorted.length) - 1
      return sorted[Math.max(0, idx)]
    }

    // Group by duration buckets
    const buckets = {
      '0-5s': entriesToUse.filter(e => (e.audio_duration ?? 0) < 5),
      '5-15s': entriesToUse.filter(e => (e.audio_duration ?? 0) >= 5 && (e.audio_duration ?? 0) < 15),
      '15-30s': entriesToUse.filter(e => (e.audio_duration ?? 0) >= 15 && (e.audio_duration ?? 0) < 30),
      '≥30s': entriesToUse.filter(e => (e.audio_duration ?? 0) >= 30),
    }

    // Calculate latency (ms) for short segments
    const latency: Record<string, { p50: number; p90: number; count: number }> = {}
    for (const [bucket, items] of Object.entries(buckets)) {
      if (bucket === '≥30s') continue
      const times = items.map(e => (e.transcription_time ?? 0) * 1000) // Convert to ms
      latency[bucket] = {
        p50: Math.round(percentile(times, 50)),
        p90: Math.round(percentile(times, 90)),
        count: items.length,
      }
    }

    // Calculate speedup for long-form (≥30s)
    const longForm = buckets['≥30s']
    const speedups = longForm.map(e => {
      const duration = e.audio_duration ?? 1
      const time = e.transcription_time ?? duration
      return duration / time
    })

    return {
      latency,
      longForm: {
        p50: speedups.length > 0 ? percentile(speedups, 50).toFixed(1) : '-',
        p90: speedups.length > 0 ? percentile(speedups, 90).toFixed(1) : '-',
        count: longForm.length,
      },
      totalSamples: entriesToUse.length,
    }
  }, [entries, filteredEntries, filterDataset])

  // Compute dataset sample index (for entries without dataset_sample_idx)
  const getDatasetSampleIdx = useCallback((entry: EvaluationEntry): number | null => {
    if (entry.dataset_sample_idx !== undefined) {
      return entry.dataset_sample_idx
    }
    // Compute from entries array - count how many entries with same dataset come before this one
    if (!entry.dataset) return null
    let idx = 0
    for (const e of entries) {
      if (e.id === entry.id) return idx
      if (e.dataset === entry.dataset) idx++
    }
    return null
  }, [entries])

  // Audio playback
  const playAudio = useCallback(() => {
    if (!selectedEntry?.dataset) return

    const sampleIdx = getDatasetSampleIdx(selectedEntry)
    if (sampleIdx === null) return

    const url = api.getAudioUrl(selectedEntry.dataset, sampleIdx)

    if (audioRef.current) {
      audioRef.current.pause()
    }

    const audio = new Audio(url)
    audioRef.current = audio

    audio.onplay = () => setIsPlaying(true)
    audio.onpause = () => setIsPlaying(false)
    audio.onended = () => setIsPlaying(false)
    audio.onerror = () => {
      setIsPlaying(false)
      setError('Failed to load audio')
    }

    audio.play()
  }, [selectedEntry, getDatasetSampleIdx])

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    setIsPlaying(false)
  }, [])

  // Clean up audio when modal closes
  useEffect(() => {
    if (!selectedEntry) {
      stopAudio()
    }
  }, [selectedEntry, stopAudio])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Run Evaluation</CardTitle>
          <CardDescription>Start a new ASR evaluation job</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-1 block">Dataset</label>
              <select
                className="w-full px-3 py-2 border rounded-md bg-background"
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={isRunning}
              >
                <option value="all">All Datasets</option>
                {datasets.map(d => (
                  <option key={d.id} value={d.id}>{d.id}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium mb-1 block">Model</label>
              <select
                className="w-full px-3 py-2 border rounded-md bg-background"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isRunning}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium mb-1 block">Max Samples</label>
              <input
                type="number"
                className="w-full px-3 py-2 border rounded-md bg-background"
                placeholder="All"
                value={maxSamples}
                onChange={(e) => setMaxSamples(e.target.value ? parseInt(e.target.value) : '')}
                disabled={isRunning}
                min={1}
              />
            </div>
          </div>
          {/* Checkpoint indicator */}
          {checkpoint && !isRunning && (
            <div className="bg-amber-50 border border-amber-200 rounded-md p-3 text-sm">
              <div className="font-medium text-amber-800">Previous run available</div>
              <div className="text-amber-700">
                {checkpoint.completed} / {checkpoint.total} samples completed
                {checkpoint.timestamp && ` (${new Date(checkpoint.timestamp).toLocaleString()})`}
              </div>
            </div>
          )}
          <div className="flex gap-2">
            {!isRunning ? (
              <>
                <Button onClick={() => startEvaluation(false)} disabled={loading || !selectedDataset}>
                  {loading ? <Loader2 className="animate-spin" /> : <Play />}
                  {checkpoint ? 'Start Fresh' : 'Start Evaluation'}
                </Button>
                {checkpoint && (
                  <Button onClick={() => startEvaluation(true)} disabled={loading} variant="outline">
                    <Play />
                    Resume ({checkpoint.completed}/{checkpoint.total})
                  </Button>
                )}
              </>
            ) : (
              <Button variant="destructive" onClick={cancelEvaluation}>
                <Square />
                Cancel
              </Button>
            )}
          </div>
          {error && (
            <div className="text-destructive text-sm bg-destructive/10 p-2 rounded">{error}</div>
          )}
        </CardContent>
      </Card>

      {/* Progress */}
      {jobStatus && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                Progress
                {currentDataset && (
                  <span className="text-sm font-normal text-muted-foreground">
                    — {currentDataset}
                  </span>
                )}
              </span>
              <span className={`text-sm px-2 py-1 rounded ${
                jobStatus.status === 'completed' ? 'bg-green-100 text-green-800' :
                jobStatus.status === 'error' ? 'bg-red-100 text-red-800' :
                jobStatus.status === 'cancelled' ? 'bg-yellow-100 text-yellow-800' :
                'bg-blue-100 text-blue-800'
              }`}>
                {jobStatus.status}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Progress value={progress} className="h-3" />
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">
                {jobStatus.completed_samples} / {jobStatus.total_samples} samples
              </span>
              <span className={`font-medium px-2 py-0.5 rounded ${getWerColor(jobStatus.current_wer)}`}>
                Avg WER: {(jobStatus.current_wer * 100).toFixed(1)}%
              </span>
            </div>
            {/* Per-dataset WER breakdown */}
            {werByDataset.length > 1 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {werByDataset.map(({ dataset, avgWer, count }) => (
                  <span
                    key={dataset}
                    className={`text-xs px-2 py-1 rounded ${getWerColor(avgWer)}`}
                  >
                    {dataset}: {(avgWer * 100).toFixed(1)}% ({count})
                  </span>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Current Sample Being Evaluated */}
      {currentEntry && isRunning && (
        <Card className="border-blue-200 bg-blue-50/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center gap-2">
              <Loader2 className="animate-spin h-4 w-4" />
              Currently Evaluating: Sample #{currentEntry.id}
              {currentEntry.dataset && (
                <span className="text-sm font-normal text-muted-foreground">
                  ({currentEntry.dataset})
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className={`p-2 rounded ${getWerColor(currentEntry.wer)}`}>
                <span className="font-medium">WER:</span> {(currentEntry.wer * 100).toFixed(1)}%
              </div>
              <div className="p-2 rounded bg-muted">
                <span className="font-medium">Duration:</span> {currentEntry.audio_duration?.toFixed(1)}s
              </div>
              <div className="p-2 rounded bg-muted">
                <span className="font-medium">Inference:</span> {currentEntry.transcription_time?.toFixed(2)}s
              </div>
            </div>
            <DiffView reference={currentEntry.reference_text} predicted={currentEntry.predicted_text} />
          </CardContent>
        </Card>
      )}

      {/* Timing Benchmarks */}
      {entries.length > 0 && timingBenchmarks.totalSamples > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Timing Benchmarks</CardTitle>
            <CardDescription>
              Latency (ms) for short segments, Speedup (x) for long-form audio
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-auto border rounded">
              <table className="w-full text-sm">
                <thead className="bg-muted">
                  <tr>
                    <th className="text-left p-2 font-medium">Duration</th>
                    <th className="text-right p-2 font-medium">Samples</th>
                    <th className="text-right p-2 font-medium">p50</th>
                    <th className="text-right p-2 font-medium">p90</th>
                  </tr>
                </thead>
                <tbody>
                  {['0-5s', '5-15s', '15-30s'].map(bucket => {
                    const data = timingBenchmarks.latency[bucket]
                    if (!data || data.count === 0) return null
                    return (
                      <tr key={bucket} className="border-t">
                        <td className="p-2 font-medium">{bucket}</td>
                        <td className="p-2 text-right text-muted-foreground">{data.count}</td>
                        <td className="p-2 text-right">{data.p50} ms</td>
                        <td className="p-2 text-right">{data.p90} ms</td>
                      </tr>
                    )
                  })}
                  {timingBenchmarks.longForm.count > 0 && (
                    <tr className="border-t bg-blue-50/50">
                      <td className="p-2 font-medium">≥30s (speedup)</td>
                      <td className="p-2 text-right text-muted-foreground">{timingBenchmarks.longForm.count}</td>
                      <td className="p-2 text-right">{timingBenchmarks.longForm.p50}x</td>
                      <td className="p-2 text-right">{timingBenchmarks.longForm.p90}x</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Short segments: transcription time in milliseconds (lower is better).
              Long-form: speedup factor vs real-time (higher is better).
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Table */}
      {entries.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span>Evaluated Samples ({filterDataset === 'all' ? entries.length : `${filteredEntries.length}/${entries.length}`})</span>
                {uniqueDatasets.length > 1 && (
                  <select
                    className="text-sm font-normal px-2 py-1 border rounded-md bg-background"
                    value={filterDataset}
                    onChange={(e) => setFilterDataset(e.target.value)}
                  >
                    <option value="all">All datasets</option>
                    {uniqueDatasets.map(ds => (
                      <option key={ds} value={ds}>{ds}</option>
                    ))}
                  </select>
                )}
              </div>
              <div className="flex gap-2 text-sm font-normal">
                <span className="px-2 py-0.5 rounded bg-green-100 text-green-800">
                  Good (&lt;10%): {filteredEntries.filter(e => e.wer < 0.1).length}
                </span>
                <span className="px-2 py-0.5 rounded bg-yellow-100 text-yellow-800">
                  Fair (10-30%): {filteredEntries.filter(e => e.wer >= 0.1 && e.wer < 0.3).length}
                </span>
                <span className="px-2 py-0.5 rounded bg-red-100 text-red-800">
                  Poor (&gt;30%): {filteredEntries.filter(e => e.wer >= 0.3).length}
                </span>
              </div>
            </CardTitle>
            {/* Per-dataset WER summary */}
            {werByDataset.length > 0 && (
              <CardDescription className="flex flex-wrap gap-2 mt-2">
                {werByDataset.map(({ dataset, avgWer, count }) => (
                  <span
                    key={dataset}
                    className={`text-xs px-2 py-1 rounded ${getWerColor(avgWer)}`}
                  >
                    <strong>{dataset}</strong>: {(avgWer * 100).toFixed(1)}% ({count} samples)
                  </span>
                ))}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent>
            <div ref={tableRef} className="max-h-96 overflow-auto border rounded">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-muted">
                  <tr>
                    <th className="text-left p-2 font-medium cursor-pointer hover:bg-muted" onClick={() => toggleSort('id')}>
                      ID<SortIcon col="id" />
                    </th>
                    <th className="text-left p-2 font-medium">Dataset</th>
                    <th className="text-left p-2 font-medium">Reference</th>
                    <th className="text-left p-2 font-medium">Predicted</th>
                    <th className="text-right p-2 font-medium cursor-pointer hover:bg-muted" onClick={() => toggleSort('wer')}>
                      WER<SortIcon col="wer" />
                    </th>
                    <th className="text-right p-2 font-medium cursor-pointer hover:bg-muted" onClick={() => toggleSort('duration')}>
                      Duration<SortIcon col="duration" />
                    </th>
                    <th className="text-right p-2 font-medium cursor-pointer hover:bg-muted" onClick={() => toggleSort('time')}>
                      Time<SortIcon col="time" />
                    </th>
                    <th className="text-right p-2 font-medium cursor-pointer hover:bg-muted" onClick={() => toggleSort('rtf')}>
                      RTF<SortIcon col="rtf" />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedEntries.map((e) => (
                    <tr
                      key={e.id}
                      className="border-t hover:bg-muted/50 cursor-pointer"
                      onClick={() => setSelectedEntry(e)}
                    >
                      <td className="p-2">{e.id}</td>
                      <td className="p-2 text-muted-foreground">{e.dataset || '-'}</td>
                      <td className="p-2 max-w-40 truncate" dir="rtl" title={e.reference_text}>
                        {e.reference_text}
                      </td>
                      <td className="p-2 max-w-40 truncate" dir="rtl" title={e.predicted_text}>
                        {e.predicted_text}
                      </td>
                      <td className="p-2 text-right">
                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${getWerColor(e.wer)}`}>
                          {(e.wer * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="p-2 text-right text-muted-foreground">
                        {e.audio_duration?.toFixed(1)}s
                      </td>
                      <td className="p-2 text-right text-muted-foreground">
                        {e.transcription_time?.toFixed(2)}s
                      </td>
                      <td className="p-2 text-right text-muted-foreground">
                        {e.audio_duration && e.transcription_time
                          ? (e.transcription_time / e.audio_duration).toFixed(2)
                          : '-'}x
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Sample Detail Modal */}
      {selectedEntry && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
          onClick={() => setSelectedEntry(null)}
        >
          <Card className="max-w-3xl w-full max-h-[85vh] overflow-auto" onClick={e => e.stopPropagation()}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Sample #{selectedEntry.id}</span>
                {selectedEntry.dataset && (
                  <span className="text-sm font-normal bg-muted px-2 py-1 rounded">
                    {selectedEntry.dataset}
                  </span>
                )}
              </CardTitle>
              <CardDescription className="flex flex-wrap gap-4 mt-2">
                <span className={`px-2 py-1 rounded ${getWerColor(selectedEntry.wer)}`}>
                  WER: {(selectedEntry.wer * 100).toFixed(1)}%
                </span>
                <span className="px-2 py-1 rounded bg-muted">
                  Duration: {selectedEntry.audio_duration?.toFixed(1)}s
                </span>
                <span className="px-2 py-1 rounded bg-muted">
                  Inference: {selectedEntry.transcription_time?.toFixed(2)}s
                </span>
                <span className="px-2 py-1 rounded bg-muted">
                  RTF: {selectedEntry.audio_duration && selectedEntry.transcription_time
                    ? (selectedEntry.transcription_time / selectedEntry.audio_duration).toFixed(2)
                    : '-'}x
                </span>
              </CardDescription>
              {/* Show locally ignored words */}
              {(() => {
                const entryKey = getEntryKey(selectedEntry)
                const ignoredWords = localIgnoredWords[entryKey] || []
                if (ignoredWords.length === 0) return null
                return (
                  <div className="flex items-center gap-2 mt-2 text-sm">
                    <span className="text-orange-600 bg-orange-50 px-2 py-1 rounded">
                      Local ignores ({ignoredWords.length}): {ignoredWords.join(', ')}
                    </span>
                    <button
                      className="text-xs text-red-600 hover:text-red-800 underline"
                      onClick={() => {
                        setLocalIgnoredWords(prev => {
                          const newState = { ...prev }
                          delete newState[entryKey]
                          return newState
                        })
                        // Recalculate without ignored words
                        api.recalculateWer([selectedEntry]).then(result => {
                          const updated = (result.entries as EvaluationEntry[])[0]
                          if (updated) {
                            setEntries(prev => prev.map(e => e.id === updated.id ? updated : e))
                            setSelectedEntry(updated)
                          }
                        })
                      }}
                    >
                      Clear
                    </button>
                  </div>
                )
              })()}
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Normalized text - this is what WER is calculated on, click to mark equivalences */}
              {selectedEntry.norm_reference_text && (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <label className="text-sm font-medium">Normalized Text (WER calculated here)</label>
                  </div>
                  <DiffView
                    reference={selectedEntry.norm_reference_text}
                    predicted={selectedEntry.norm_predicted_text || ''}
                    onMarkEquivalent={handleMarkEquivalent}
                    onIgnoreWord={handleIgnoreWord}
                    onIgnoreWordGlobal={handleIgnoreWordGlobal}
                  />
                </div>
              )}
              {/* Original text - just for reference */}
              <div>
                <label className="text-sm font-medium mb-2 block text-muted-foreground">Original Text (for reference)</label>
                <DiffView
                  reference={selectedEntry.reference_text}
                  predicted={selectedEntry.predicted_text}
                />
              </div>
              <div className="flex justify-between items-center pt-2">
                {/* Audio playback */}
                <div>
                  {selectedEntry.dataset && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={isPlaying ? stopAudio : playAudio}
                    >
                      {isPlaying ? (
                        <>
                          <Pause className="h-4 w-4 mr-1" />
                          Stop
                        </>
                      ) : (
                        <>
                          <Volume2 className="h-4 w-4 mr-1" />
                          Play Audio
                        </>
                      )}
                    </Button>
                  )}
                </div>
                <Button variant="outline" onClick={() => setSelectedEntry(null)}>
                  Close
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
