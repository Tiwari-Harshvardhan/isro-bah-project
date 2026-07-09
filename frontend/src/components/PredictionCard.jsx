const PredictionCard = ({ prediction, loading, selectedZone }) => {
  if (!selectedZone) {
    return (
      <div className="card prediction-card placeholder-card">
        <div className="card-title-row">
          <h3>Prediction</h3>
        </div>
        <p className="placeholder-text">Select a Delhi zone on the map or search above to view heat temperature prediction.</p>
      </div>
    )
  }

  if (loading || !prediction) {
    return (
      <div className="card skeleton-card">
        <div className="skeleton-line" />
        <div className="skeleton-line short" />
        <div className="skeleton-line" />
      </div>
    )
  }

  return (
    <div className="card prediction-card">
      <div className="card-title-row">
        <h3>Prediction</h3>
        <span className="risk-level-badge">{prediction.risk_level || 'Unknown'} Risk</span>
      </div>
      
      <div className="prediction-main">
        <h1>{prediction.predicted_lst?.toFixed(2) || 0}°C</h1>
        <p>Predicted Land Surface Temperature</p>

        {/* Prediction Verification Status Badge */}
        <div className="verification-badge-container">
          {prediction.verification_status === 'Verified' ? (
            <span className="badge-verified">Verified ✓</span>
          ) : (
            <span className="badge-needs-review">Needs Review ⚠</span>
          )}
          {prediction.verification_details && (
            <p className="verification-details-text">{prediction.verification_details}</p>
          )}
        </div>
      </div>

      <div className="stats-grid prediction-grid">
        <div>
          <p>Historical LST</p>
          <strong>{prediction.historical_lst?.toFixed(2) || 0}°C</strong>
        </div>
        <div>
          <p>Population</p>
          <strong>{prediction.population?.toLocaleString() || 0}</strong>
        </div>
        <div>
          <p>Density</p>
          <strong>{prediction.population_density?.toLocaleString() || 0}/km²</strong>
        </div>
        <div>
          <p>Built-up %</p>
          <strong>{prediction.built_up_percent?.toFixed(2) || 0}%</strong>
        </div>
        <div>
          <p>NDVI</p>
          <strong>{prediction.mean_ndvi?.toFixed(3) || 0}</strong>
        </div>
      </div>

      <div className="recommendations">
        <h4>Dynamic Recommendations</h4>
        <div className="recommendation-list">
          {(prediction.recommendation || []).map((item, index) => {
            if (typeof item === 'object') {
              return (
                <div key={index} className="recommendation-item-card">
                  <div className="rec-header">
                    <strong>{item.title}</strong>
                    <span className={`priority-pill ${item.priority?.toLowerCase()}`}>{item.priority}</span>
                  </div>
                  <p className="rec-desc">{item.description}</p>
                  <div className="rec-footer">
                    <span>Impact: <strong>{item.expected_cooling_impact}</strong></span>
                    <span>Cost: <strong>{item.estimated_implementation_cost}</strong></span>
                    <span>Pop: <strong>{item.affected_population?.toLocaleString()}</strong></span>
                  </div>
                </div>
              )
            } else {
              return <div key={index} className="recommendation-item-simple">{item}</div>
            }
          })}
        </div>
      </div>
    </div>
  )
}

export default PredictionCard
