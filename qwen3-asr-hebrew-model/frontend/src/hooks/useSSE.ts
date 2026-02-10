import { useCallback, useEffect, useRef, useState } from 'react'
import type { SSEProgressEvent } from '@/api/types'
import { api } from '@/api/client'

interface UseSSEOptions {
  onProgress?: (event: SSEProgressEvent) => void
  onComplete?: (event: SSEProgressEvent) => void
  onError?: (error: string) => void
}

export function useSSE(jobId: string | null, options: UseSSEOptions = {}) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<SSEProgressEvent | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  // Use refs for callbacks to avoid re-creating EventSource on callback changes
  const onProgressRef = useRef(options.onProgress)
  const onCompleteRef = useRef(options.onComplete)
  const onErrorRef = useRef(options.onError)

  // Keep refs up to date
  useEffect(() => {
    onProgressRef.current = options.onProgress
    onCompleteRef.current = options.onComplete
    onErrorRef.current = options.onError
  }, [options.onProgress, options.onComplete, options.onError])

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
      setIsConnected(false)
    }
  }, [])

  useEffect(() => {
    if (!jobId) {
      disconnect()
      return
    }

    // Don't reconnect if already connected to same job
    if (eventSourceRef.current) {
      return
    }

    const url = api.getStreamUrl(jobId)
    console.log('SSE connecting to:', url)
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      console.log('SSE connected')
      setIsConnected(true)
    }

    eventSource.onmessage = (event) => {
      try {
        const data: SSEProgressEvent = JSON.parse(event.data)
        console.log('SSE event:', data.event, data)
        setLastEvent(data)

        if (data.event === 'progress' || data.event === 'started') {
          onProgressRef.current?.(data)
        } else if (data.event === 'complete') {
          onCompleteRef.current?.(data)
          disconnect()
        } else if (data.event === 'error') {
          onErrorRef.current?.(data.error || 'Unknown error')
          disconnect()
        } else if (data.event === 'cancelled') {
          disconnect()
        }
      } catch (e) {
        console.error('Failed to parse SSE event:', e)
      }
    }

    eventSource.onerror = (e) => {
      console.error('SSE error:', e)
      setIsConnected(false)
      onErrorRef.current?.('Connection lost')
    }

    return () => {
      disconnect()
    }
  }, [jobId, disconnect])

  return { isConnected, lastEvent, disconnect }
}
