import React, { useState } from 'react';
import axios from 'axios';
import { Activity, AlertTriangle, CheckCircle, Brain, HeartPulse } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { API_BASE_URL } from './config';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    age: '',
    hypertension: 0,
    heart_disease: 0,
    avg_glucose_level: '',
    bmi: '',
    gender: 'Male',
    work_type: 'Private',
    smoking_status: 'never smoked',
    residence_type: 'Urban',
    ever_married: 'Yes'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        age: parseFloat(formData.age) || 0,
        hypertension: parseInt(formData.hypertension),
        heart_disease: parseInt(formData.heart_disease),
        avg_glucose_level: parseFloat(formData.avg_glucose_level) || 0,
        bmi: parseFloat(formData.bmi) || 0,
        gender: formData.gender,
        work_type: formData.work_type,
        smoking_status: formData.smoking_status,
        residence_type: formData.residence_type,
        ever_married: formData.ever_married
      };

      const res = await axios.post(`${API_BASE_URL}/predict`, payload);
      setResult(res.data);
    } catch (err) {
      setError("Failed to connect to the prediction engine. Ensure API is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Recharts Gauge logic
  const renderGauge = (probability) => {
    const val = probability * 100;
    const data = [
      { name: 'Risk', value: val, color: val >= 30 ? '#ef4444' : '#10b981' },
      { name: 'Safe', value: 100 - val, color: 'rgba(255,255,255,0.05)' }
    ];

    return (
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="100%"
            startAngle={180}
            endAngle={0}
            innerRadius={80}
            outerRadius={110}
            dataKey="value"
            stroke="none"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    );
  };

  const renderShapContributions = (contributions) => {
    if (!contributions || contributions.length === 0) return null;
    return (
      <div className="shap-contributions" style={{ marginTop: '24px' }}>
        <h4 style={{ marginBottom: '8px', color: 'var(--text-primary)' }}>Feature Contributions</h4>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {contributions.map((c, idx) => (
            <li key={idx} style={{ marginBottom: '4px', color: 'var(--text-secondary)' }}>
              <strong>{c.feature}</strong>: {c.contribution.toFixed(3)} ({c.direction})
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="app-layout">
      <div className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '40px', padding: '24px 16px 0 16px' }}>
          <div style={{ background: 'var(--blue-bg)', padding: '8px', borderRadius: '8px', display: 'flex', color: 'var(--blue)' }}>
            <Brain size={24} />
          </div>
          <h2 className="text-heading-1" style={{ margin: 0, color: '#ffffffff' }}>Prediction Engine</h2>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="sidebar-group-label">Menu</div>
          <div className="sidebar-nav-item active">
            <div className="sidebar-nav-icon"><Activity size={18} /></div>
            <div className="sidebar-nav-label">New Prediction</div>
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="content-max">
          <div className="zone-a">
            <h1 className="text-hero" style={{ marginBottom: '8px' }}>Stroke Risk Assessment</h1>
            <p className="text-body" style={{ color: 'var(--text-secondary)', maxWidth: '800px' }}>Enter patient clinical details to generate an AI-powered stroke probability assessment using the 64-node baseline network.</p>
          </div>

          <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
            {/* Form Section */}
            <form onSubmit={handlePredict} className="card-type-5" style={{ flex: '1 1 600px' }}>
              <h3 className="text-heading-1" style={{ marginBottom: '24px' }}>Patient Details</h3>
              <div className="form-grid">
                <div className="input-group">
                  <label>Age (Years)</label>
                  <input required type="number" step="0.1" name="age" value={formData.age} onChange={handleChange} className="glass-input" placeholder="e.g. 65" />
                </div>
                <div className="input-group">
                  <label>Average Glucose Level</label>
                  <input required type="number" step="0.1" name="avg_glucose_level" value={formData.avg_glucose_level} onChange={handleChange} className="glass-input" placeholder="e.g. 105.2" />
                </div>
                <div className="input-group">
                  <label>BMI</label>
                  <input required type="number" step="0.1" name="bmi" value={formData.bmi} onChange={handleChange} className="glass-input" placeholder="e.g. 28.5" />
                </div>
                <div className="input-group">
                  <label>Gender</label>
                  <select name="gender" value={formData.gender} onChange={handleChange} className="glass-input">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div className="form-grid">
                <div className="input-group">
                  <label>Hypertension</label>
                  <select name="hypertension" value={formData.hypertension} onChange={handleChange} className="glass-input">
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>Heart Disease</label>
                  <select name="heart_disease" value={formData.heart_disease} onChange={handleChange} className="glass-input">
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>Smoking Status</label>
                  <select name="smoking_status" value={formData.smoking_status} onChange={handleChange} className="glass-input">
                    <option value="never smoked">Never Smoked</option>
                    <option value="formerly smoked">Formerly Smoked</option>
                    <option value="smokes">Smokes</option>
                    <option value="Unknown">Unknown</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>Work Type</label>
                  <select name="work_type" value={formData.work_type} onChange={handleChange} className="glass-input">
                    <option value="Private">Private</option>
                    <option value="Self-employed">Self-employed</option>
                    <option value="Govt_job">Govt Job</option>
                    <option value="children">Children</option>
                    <option value="Never_worked">Never Worked</option>
                  </select>
                </div>
              </div>

              <div className="form-grid">
                <div className="input-group">
                  <label>Residence Type</label>
                  <select name="residence_type" value={formData.residence_type} onChange={handleChange} className="glass-input">
                    <option value="Urban">Urban</option>
                    <option value="Rural">Rural</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>Ever Married</label>
                  <select name="ever_married" value={formData.ever_married} onChange={handleChange} className="glass-input">
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
              </div>

              <button type="submit" className="primary-btn" disabled={loading}>
                {loading ? <><span className="spinner"></span> Analyzing...</> : "Generate Prediction"}
              </button>
              {error && <div style={{ color: 'var(--red)', marginTop: '12px', fontSize: '14px' }}>{error}</div>}
            </form>

            {/* Result Section */}
            <div className="card-type-5" style={{ flex: '1 1 350px', display: 'flex', flexDirection: 'column' }}>
              <div className="t5-header">
                <h3 className="text-heading-1" style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
                  <HeartPulse size={20} color="var(--blue)" /> Results Dashboard
                </h3>
              </div>

              {!result && !loading && (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', textAlign: 'center', padding: '40px 0' }}>
                  Awaiting patient data to run inference.
                </div>
              )}

              {loading && (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div className="spinner"></div>
                </div>
              )}

              {result && !loading && (
                <div className="results-container">
                  <div className="gauge-container" style={{ position: 'relative' }}>
                    {renderGauge(result.probability)}
                    <div style={{ position: 'absolute', bottom: '20px', textAlign: 'center' }}>
                      <div style={{ fontSize: '48px', fontWeight: 'bold', lineHeight: 1 }}>{(result.probability * 100).toFixed(1)}<span style={{ fontSize: '24px', color: 'var(--text-secondary)' }}>%</span></div>
                      <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Probability</div>
                    </div>
                  </div>

                  {result.risk_level === "High" ? (
                    <div className="risk-alert high">
                      <AlertTriangle size={24} />
                      <div>
                        <div style={{ fontSize: '16px' }}>High Stroke Risk Detected</div>
                        <div style={{ fontSize: '13px', fontWeight: 'normal', opacity: 0.9 }}>Probability exceeds the {result.threshold * 100}% optimal threshold.</div>
                      </div>
                    </div>
                  ) : (
                    <div className="risk-alert low">
                      <CheckCircle size={24} />
                      <div>
                        <div style={{ fontSize: '16px' }}>Low Stroke Risk</div>
                        <div style={{ fontSize: '13px', fontWeight: 'normal', opacity: 0.9 }}>Probability falls below the {result.threshold * 100}% optimal threshold.</div>
                      </div>
                    </div>
                  )}
                  {renderShapContributions(result.shap_contributions)}
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
