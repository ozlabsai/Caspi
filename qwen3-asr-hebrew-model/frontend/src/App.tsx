import { useState } from 'react'
import { EvaluationPanel } from '@/components/EvaluationPanel'
import { ResultsPanel } from '@/components/ResultsPanel'
import './index.css'

function App() {
  const [activeTab, setActiveTab] = useState<'evaluate' | 'results'>('evaluate')

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold">Hebrew ASR Evaluation</h1>
          <p className="text-muted-foreground">Qwen3-ASR Benchmark Dashboard</p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {/* Tab buttons */}
        <div className="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground mb-4">
          <button
            onClick={() => setActiveTab('evaluate')}
            className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all ${
              activeTab === 'evaluate' ? 'bg-background text-foreground shadow-sm' : ''
            }`}
          >
            Run Evaluation
          </button>
          <button
            onClick={() => setActiveTab('results')}
            className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all ${
              activeTab === 'results' ? 'bg-background text-foreground shadow-sm' : ''
            }`}
          >
            View Results
          </button>
        </div>

        {/* Both panels always mounted, just hidden with CSS */}
        <div className={activeTab === 'evaluate' ? '' : 'hidden'}>
          <EvaluationPanel />
        </div>
        <div className={activeTab === 'results' ? '' : 'hidden'}>
          <ResultsPanel />
        </div>
      </main>
    </div>
  )
}

export default App
