import { useState } from 'react'

const BudgetPlanner = ({ budget, onAllocate, budgetResult }) => {
  const [amount, setAmount] = useState(budget || '')

  return (
    <div className="card budget-card">
      <div className="card-title-row">
        <h3>Budget Planner</h3>
        <span>Allocate funds across Delhi zones</span>
      </div>
      
      <div className="budget-input-row">
        <input
          type="text"
          value={amount}
          onChange={(event) => setAmount(event.target.value)}
          placeholder="₹5 Crore or 50000000"
          className="budget-input"
        />
        <button className="primary-button" onClick={() => onAllocate(amount)}>
          Allocate
        </button>
      </div>

      <div className="budget-summary">
        {budgetResult ? (
          <p>Proposed allocation plan for <strong>₹{budgetResult.budget?.toLocaleString()}</strong>:</p>
        ) : (
          <p className="placeholder-text">Enter a budget (e.g. ₹5 Crore) to build a heat mitigation plan.</p>
        )}
      </div>

      {budgetResult && budgetResult.priority_summary && (
        <div className="budget-results-container">
          <div className="table-responsive">
            <table className="budget-table">
              <thead>
                <tr>
                  <th>Zone</th>
                  <th>Budget</th>
                  <th>Population Benefited</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                {budgetResult.priority_summary.map((item, index) => (
                  <tr key={index}>
                    <td>{item.zone}</td>
                    <td>₹{(item.suggested_budget / 10000000).toFixed(2)} Cr</td>
                    <td>{item.expected_population?.toLocaleString()}</td>
                    <td>
                      <span className={`priority-pill ${item.risk_level?.toLowerCase()}`}>
                        {item.risk_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="budget-rationale-box">
            <h4>Allocation Rationale</h4>
            <p>
              Funds are distributed dynamically using a multi-criteria optimization algorithm. 
              The score considers: <strong>Heat Severity (LST)</strong>, <strong>Population Density</strong>, 
              <strong>Built-up area percentage</strong>, and <strong>Vegetation Deficit (1 - NDVI)</strong>. 
              High-severity, dense zones with low tree canopy are prioritized to maximize cooling impact.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default BudgetPlanner
