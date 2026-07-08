const ExplanationPanel = ({ analysis, loading }) => {
  if (loading) {
    return <div className="card skeleton-card"><div className="skeleton-line" /><div className="skeleton-line short" /><div className="skeleton-line" /></div>
  }

  return (
    <div className="card explanation-card">
      <div className="card-title-row">
        <h3>AI Explanation</h3>
        <span>{analysis?.confidence ? `${analysis.confidence}% confidence` : 'Awaiting prediction'}</span>
      </div>
      <div className="explanation-grid">
        <div><p>Reasoning</p><ul>{analysis?.reasoning?.map((item, idx) => <li key={idx}>{item}</li>)}</ul></div>
        <div><p>Historical Comparison</p><p>{analysis?.historical_comparison || 'No data available.'}</p></div>
        <div><p>Neighbour Comparison</p><p>{analysis?.neighbour_comparison || 'No data available.'}</p></div>
        <div><p>Anomalies</p><ul>{analysis?.potential_anomalies?.map((item, idx) => <li key={idx}>{item}</li>)}</ul></div>
      </div>
    </div>
  )
}

export default ExplanationPanel
