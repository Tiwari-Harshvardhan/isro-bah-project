import { useEffect, useMemo, useState } from 'react'

const AIChatPanel = ({ analysis, onPrompt, selectedZone, defaultResponse }) => {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)

  // Sync default responses from the parent (e.g. when budget allocation occurs)
  useEffect(() => {
    if (defaultResponse) {
      setResponse(defaultResponse)
    }
  }, [defaultResponse])

  const handleAsk = async () => {
    if (!query.trim()) return
    setLoading(true)
    const answer = await onPrompt(query)
    setResponse(answer)
    setLoading(false)
  }

  const answerText = useMemo(() => {
    if (!response) return 'Ask a question about zone heat, budgets, or mitigation.'
    return response
  }, [response])

  if (!selectedZone) {
    return (
      <div className="card ai-panel placeholder-card">
        <div className="card-title-row">
          <h3>AI Planning Assistant</h3>
        </div>
        <p className="placeholder-text">Select a Delhi zone to activate the AI Planning Assistant.</p>
      </div>
    )
  }

  return (
    <div className="card ai-panel">
      <div className="card-title-row">
        <h3>AI Planning Assistant</h3>
        <span>Ask for insights or budget planning</span>
      </div>
      <div className="ai-summary">
        <p><strong>Prediction</strong>: {analysis?.prediction ?? '--'}°C</p>
        <p><strong>Confidence</strong>: {analysis?.confidence ?? '--'}%</p>
      </div>
      <textarea
        rows={4}
        className="chat-input"
        value={query}
        disabled={loading}
        placeholder="Ask the assistant: 'What caused high temperature?', 'How many trees should be planted?', 'What interventions help?'"
        onChange={(event) => setQuery(event.target.value)}
      />
      <div className="ai-action-row">
        <button className="primary-button" onClick={handleAsk} disabled={loading || !query.trim()}>
          {loading ? 'Thinking…' : 'Ask Assistant'}
        </button>
      </div>
      <div className="chat-response">
        <p>{answerText}</p>
      </div>
    </div>
  )
}

export default AIChatPanel
