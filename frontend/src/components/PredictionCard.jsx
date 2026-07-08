const PredictionCard = ({ prediction, loading }) => {
  if (loading) {
    return <div className="card skeleton-card"><div className="skeleton-line" /><div className="skeleton-line short" /><div className="skeleton-line" /></div>
  }

  return (
    <div className="card prediction-card">
      <div className="card-title-row">
        <h3>Prediction</h3>
        <span>{prediction?.risk_level || 'Unknown'}</span>
      </div>
      <div className="prediction-main">
        <h1>{prediction?.predicted_lst?.toFixed(2) || 0}°C</h1>
        <p>Predicted temperature</p>
      </div>
      <div className="stats-grid prediction-grid">
        <div><p>Historical LST</p><strong>{prediction?.historical_lst?.toFixed(2) || 0}°C</strong></div>
        <div><p>Population</p><strong>{prediction?.population?.toFixed(0) || 0}</strong></div>
        <div><p>Density</p><strong>{prediction?.population_density?.toFixed(0) || 0}</strong></div>
        <div><p>Built-up %</p><strong>{prediction?.built_up_percent?.toFixed(2) || 0}</strong></div>
        <div><p>NDVI</p><strong>{prediction?.mean_ndvi?.toFixed(3) || 0}</strong></div>
      </div>
      <div className="recommendations">
        <h4>Recommendations</h4>
        <ul>
          {(prediction?.recommendation || []).map((item, index) => <li key={index}>{item}</li>)}
        </ul>
      </div>
    </div>
  )
}

export default PredictionCard
