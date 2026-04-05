import { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader2, Database, ShieldAlert, Award, Activity, ActivitySquare, Dna, FileEdit } from 'lucide-react';
import './index.css';
import { MolecularViewer } from './components/MolecularViewer';

interface Target {
  name: string;
  pdb_id: string;
  url: string;
}

interface Prediction {
  drug_name: string;
  gnn_score: number;
  affinity: number;
  // agreement: 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR';
  // agreement_score: number;
  ligand_url?: string;
}

interface ResultsResponse {
  disease: string;
  targets: Target[];
  predictions: Prediction[];
}

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState('');

  // State to track which 3D viewers are open
  const [openViewers, setOpenViewers] = useState<Record<string, boolean>>({});
  const [activeTarget, setActiveTarget] = useState<Target | null>(null);

  const toggleViewer = (id: string) => {
    setOpenViewers(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setResults(null);
    setOpenViewers({});
    setActiveTarget(null);

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_URL}/predict`, {
        disease: query,
        top_k: 12
      });
      setResults(response.data);
      if (response.data.targets && response.data.targets.length > 0) {
        setActiveTarget(response.data.targets[0])
      }
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
        'Failed to connect to the prediction server. Please ensure the backend is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  // const getAgreementClass = (agreement: string) => {
  //   switch (agreement) {
  //     case 'EXCELLENT': return 'excellent';
  //     case 'GOOD': return 'good';
  //     case 'FAIR': return 'fair';
  //     case 'POOR': return 'poor';
  //     default: return 'poor';
  //   }
  // };
  const getAgreementClass = (affinity: number) => {
    if (affinity <= -8) {
      return { label: "Strong Candidate for Research", class: "excellent" };
    }
    if (affinity <= -7) {
      return { label: "Promising Interaction", class: "good" };
    }
    if (affinity <= -6) {
      return { label: "Moderate Interaction", class: "fair" };
    }
    return { label: "Weak Interaction", class: "poor" };
  };

  const getScoreColor = (score: number) => {
    if (score > 0.8) return 'var(--success)';
    if (score > 0.6) return 'var(--primary)';
    if (score > 0.4) return 'var(--warning)';
    return 'var(--danger)';
  };

  return (
    <div className="app-container">
      <motion.header
        className="header"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <h1>Drug Repurposing AI</h1>
        <p>Discover novel therapeutic candidates using our advanced Graph Neural Network trained on the PrimeKG biomedical knowledge graph.</p>
      </motion.header>

      <motion.div
        className="search-container"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <form onSubmit={handleSearch} className="search-box">
          <input
            type="text"
            className="search-input"
            placeholder="Search for a disease (e.g., Anemia, Parkinson...)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="search-btn" disabled={!query.trim() || loading}>
            {loading ? <Loader2 className="animate-spin" /> : <Search />}
            {loading ? 'Analyzing...' : 'Discover'}
          </button>
        </form>
      </motion.div>

      <AnimatePresence mode="wait">
        {loading && (
          <motion.div
            key="loading"
            className="loading-container"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="loader"></div>
            <div className="loading-text">Simulating molecular interactions across the knowledge graph...</div>
          </motion.div>
        )}

        {error && (
          <motion.div
            key="error"
            className="error-message"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <ShieldAlert style={{ marginBottom: '1rem', display: 'inline-block' }} size={32} />
            <p>{error}</p>
          </motion.div>
        )}

        {!loading && !error && !results && (
          <motion.div
            key="empty"
            className="empty-state"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <Database size={64} />
            <h2>Ready for Analysis</h2>
            <p>Enter a disease above to find potential repurposed drug candidates.</p>
          </motion.div>
        )}

        {results && (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="results-container"
          >
            <div className="results-header" style={{ marginBottom: '2rem' }}>
              <h2 className="results-title">
                Analysis Results for <span className="results-disease-name">{results.disease}</span>
              </h2>
            </div>

            {/* Target Proteins Section */}
            {results.targets && results.targets.length > 0 && activeTarget && (
              <div style={{ marginBottom: '3rem', padding: '1.5rem', background: 'rgba(30, 41, 59, 0.5)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: '#e2e8f0', fontSize: '1.2rem' }}>
                  <Dna size={20} color="#3b82f6" /> Disease Target Proteins
                </h3>

                <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                  {results.targets.map(target => (
                    <button
                      key={target.name}
                      onClick={() => setActiveTarget(target)}
                      style={{
                        padding: '0.5rem 1rem',
                        background: activeTarget.name === target.name ? '#3b82f6' : 'rgba(255,255,255,0.1)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                    >
                      {target.name} ({target.pdb_id})
                    </button>
                  ))}
                </div>

                <MolecularViewer
                  url={activeTarget.url}
                  format="pdb"
                  title={`Protein: ${activeTarget.name} (PDB: ${activeTarget.pdb_id})`}
                  className="fade-in"
                />
              </div>
            )}

            <div className="cards-grid">
              {results.predictions.map((pred, idx) => (
                <motion.div
                  key={pred.drug_name}
                  className="drug-card"
                  initial={{ y: 50, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                >
                  <div className={`rank-badge rank-${idx + 1}`}>
                    {idx + 1}
                  </div>

                  <h3 className="drug-name">{pred.drug_name}</h3>

                  <div className="metrics-container">
                    {/* GNN Score */}
                    <div className="metric">
                      <div className="metric-header">
                        <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <Activity size={14} /> GNN Prediction Score
                        </span>
                        <span className="metric-value">{pred.gnn_score.toFixed(4)}</span>
                      </div>
                      <div className="progress-bg">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${pred.gnn_score * 100}%`,
                            background: getScoreColor(pred.gnn_score)
                          }}
                        ></div>
                      </div>
                    </div>

                    {/* Binding Affinity */}
                    <div className="metric">
                      <div className="metric-header">
                        <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <ActivitySquare size={14} /> Docking Affinity
                        </span>
                        <span className="metric-value">{pred.affinity.toFixed(2)} kcal/mol</span>
                      </div>
                      <div className="progress-bg">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${Math.max(0, Math.min(100, ((-pred.affinity) / 10) * 100))}%`,
                            background: '#ec4899'
                          }}
                        ></div>
                      </div>
                    </div>

                    {/* Agreement Status */}
                    <div className="metric">
                      <div className="metric-header">
                        <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <Award size={14} /> Research Potential
                        </span>
                        {(() => {
                          const research = getAgreementClass(pred.affinity);
                          return (
                            <span className={`badge ${research.class}`}>
                              {research.label}
                            </span>
                          );
                        })()}
                      </div>
                    </div>

                    {/* 3D Structure Viewer Toggle */}
                    {pred.ligand_url && (
                      <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                        <button
                          onClick={() => toggleViewer(pred.drug_name)}
                          style={{
                            width: '100%',
                            padding: '0.5rem',
                            background: 'rgba(59, 130, 246, 0.1)',
                            color: '#60a5fa',
                            border: '1px solid rgba(59, 130, 246, 0.2)',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '0.5rem',
                            transition: 'all 0.2s',
                            fontSize: '0.9rem'
                          }}
                        >
                          <FileEdit size={16} />
                          {openViewers[pred.drug_name] ? 'Hide 3D Structure' : 'View 3D Structure'}
                        </button>

                        <AnimatePresence>
                          {openViewers[pred.drug_name] && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              style={{ overflow: 'hidden', marginTop: '1rem' }}
                            >
                              <MolecularViewer
                                url={pred.ligand_url}
                                format="sdf"
                                title={pred.drug_name}
                              />
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
