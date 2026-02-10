import { useState, useEffect } from 'react'
import { FileText, RefreshCw } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { api } from '@/api/client'
import type { ResultFile, ResultData, EvaluationEntry } from '@/api/types'

export function ResultsPanel() {
  const [files, setFiles] = useState<ResultFile[]>([])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [resultData, setResultData] = useState<ResultData | null>(null)
  const [selectedEntry, setSelectedEntry] = useState<EvaluationEntry | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadFiles = async () => {
    setLoading(true)
    try {
      const f = await api.getResults()
      setFiles(f)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadFiles()
  }, [])

  useEffect(() => {
    if (!selectedFile) {
      setResultData(null)
      return
    }
    setLoading(true)
    api.getResult(selectedFile)
      .then(setResultData)
      .catch(e => setError((e as Error).message))
      .finally(() => setLoading(false))
  }, [selectedFile])

  // Calculate WER distribution for histogram
  const werDistribution = resultData ? (() => {
    const buckets = [
      { name: '0-10%', min: 0, max: 0.1, count: 0 },
      { name: '10-20%', min: 0.1, max: 0.2, count: 0 },
      { name: '20-30%', min: 0.2, max: 0.3, count: 0 },
      { name: '30-50%', min: 0.3, max: 0.5, count: 0 },
      { name: '50%+', min: 0.5, max: 1.1, count: 0 },
    ]
    resultData.rows.forEach(r => {
      const bucket = buckets.find(b => r.wer >= b.min && r.wer < b.max)
      if (bucket) bucket.count++
    })
    return buckets
  })() : []

  const getBarColor = (name: string) => {
    if (name === '0-10%') return '#22c55e'
    if (name === '10-20%') return '#84cc16'
    if (name === '20-30%') return '#eab308'
    if (name === '30-50%') return '#f97316'
    return '#ef4444'
  }

  return (
    <div className="space-y-4">
      {/* File Selector */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Evaluation Results</span>
            <Button variant="ghost" size="icon" onClick={loadFiles}>
              <RefreshCw className={loading ? 'animate-spin' : ''} />
            </Button>
          </CardTitle>
          <CardDescription>Browse completed evaluation results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {files.map(f => (
              <button
                key={f.filename}
                className={`flex items-center gap-2 p-3 rounded-md border text-left hover:bg-muted transition-colors ${
                  selectedFile === f.filename ? 'border-primary bg-muted' : ''
                }`}
                onClick={() => setSelectedFile(f.filename)}
              >
                <FileText className="w-4 h-4 text-muted-foreground" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">{f.filename}</div>
                  <div className="text-xs text-muted-foreground">
                    {f.samples} samples {f.wer !== null && `• WER: ${(f.wer * 100).toFixed(1)}%`}
                  </div>
                </div>
              </button>
            ))}
          </div>
          {error && <div className="text-destructive text-sm mt-2">{error}</div>}
        </CardContent>
      </Card>

      {resultData && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Total Samples</CardDescription>
                <CardTitle className="text-2xl">{resultData.summary.total_samples}</CardTitle>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Average WER</CardDescription>
                <CardTitle className={`text-2xl ${
                  (resultData.summary.avg_wer ?? 0) < 0.2 ? 'text-green-600' :
                  (resultData.summary.avg_wer ?? 0) < 0.4 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {resultData.summary.avg_wer !== null ? `${(resultData.summary.avg_wer * 100).toFixed(1)}%` : '-'}
                </CardTitle>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Total Audio</CardDescription>
                <CardTitle className="text-2xl">
                  {resultData.summary.total_audio_duration !== null
                    ? `${(resultData.summary.total_audio_duration / 60).toFixed(1)}m`
                    : '-'}
                </CardTitle>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Real-Time Factor</CardDescription>
                <CardTitle className="text-2xl">
                  {resultData.summary.rtf !== null ? resultData.summary.rtf.toFixed(2) : '-'}x
                </CardTitle>
              </CardHeader>
            </Card>
          </div>

          {/* WER Distribution Chart */}
          <Card>
            <CardHeader>
              <CardTitle>WER Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={werDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" name="Samples">
                      {werDistribution.map((entry) => (
                        <Cell key={entry.name} fill={getBarColor(entry.name)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Results Table */}
          <Card>
            <CardHeader>
              <CardTitle>Sample Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-96 overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-background">
                    <tr className="border-b">
                      <th className="text-left p-2">ID</th>
                      <th className="text-left p-2">Reference</th>
                      <th className="text-left p-2">Predicted</th>
                      <th className="text-right p-2">WER</th>
                      <th className="text-right p-2">Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resultData.rows.map((e) => (
                      <tr
                        key={e.id}
                        className="border-b hover:bg-muted/50 cursor-pointer"
                        onClick={() => setSelectedEntry(e)}
                      >
                        <td className="p-2">{e.id}</td>
                        <td className="p-2 max-w-48 truncate" dir="rtl">{e.reference_text}</td>
                        <td className="p-2 max-w-48 truncate" dir="rtl">{e.predicted_text}</td>
                        <td className={`p-2 text-right ${
                          e.wer < 0.1 ? 'text-green-600' : e.wer < 0.3 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {(e.wer * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-right">{e.audio_duration?.toFixed(1)}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Sample Detail Modal */}
          {selectedEntry && (
            <div
              className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
              onClick={() => setSelectedEntry(null)}
            >
              <Card className="max-w-2xl w-full max-h-[80vh] overflow-auto" onClick={e => e.stopPropagation()}>
                <CardHeader>
                  <CardTitle>Sample #{selectedEntry.id}</CardTitle>
                  <CardDescription>
                    WER: {(selectedEntry.wer * 100).toFixed(1)}% |
                    Duration: {selectedEntry.audio_duration?.toFixed(1)}s |
                    Transcription: {selectedEntry.transcription_time?.toFixed(2)}s
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm font-medium mb-1">Reference</div>
                    <div className="p-3 bg-muted rounded-md" dir="rtl">
                      {selectedEntry.reference_text}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium mb-1">Predicted</div>
                    <div className="p-3 bg-muted rounded-md" dir="rtl">
                      {selectedEntry.predicted_text}
                    </div>
                  </div>
                  <Button variant="outline" onClick={() => setSelectedEntry(null)}>
                    Close
                  </Button>
                </CardContent>
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  )
}
