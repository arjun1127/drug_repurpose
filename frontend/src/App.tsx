import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, Loader2, Database, ShieldAlert,
  Activity, Dna, FileEdit,
  BarChart3, Shield, TrendingUp, Layers,
  CheckCircle, AlertTriangle, Zap, Target,
  GitBranch
} from 'lucide-react';
import './index.css';
import { MolecularViewer } from './components/MolecularViewer';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ─── Types ────────────────────────────────────────────

interface TargetProtein {
  name: string;
  pdb_id: string;
  url: string;
}

interface Prediction {
  drug_name: string;
  gnn_score: number;
  degree: number;
  degree_bucket: 'low' | 'medium' | 'high' | 'unknown';
  ligand_url?: string;
}

interface ResultsResponse {
  disease: string;
  targets: TargetProtein[];
  predictions: Prediction[];
}

interface TrainingMetrics {
  config: Record<string, any>;
  splits: Record<string, number>;
  best: Record<string, number>;
  test: Record<string, number>;
  degree_stratified: {
    thresholds: { q33: number; q66: number };
    low: { count: number; positive_rate: number; auc: number; ap: number };
    medium: { count: number; positive_rate: number; auc: number; ap: number };
    high: { count: number; positive_rate: number; auc: number; ap: number };
  };
  bias: {
    spearman_rho: number;
    spearman_p_value: number;
    sampled_diseases: number;
    mean_jaccard: number;
    median_jaccard: number;
    p90_jaccard: number;
    pairs_compared: number;
    top1_mode_fraction: number;
    top1_mode_node: number;
    top1_mode_drug_name: string;
  };
  history: Record<string, number[]>;
}

type Tab = 'discover' | 'performance' | 'bias';

// ─── Plot Metadata ────────────────────────────────────

const PLOT_INFO: Record<string, { title: string; desc: string }> = {
  'training_curves.png': {
    title: 'Training Curves',
    desc: 'Train/validation loss and MRR progression over epochs. Early stopping triggered at the best validation MRR.'
  },
  'degree_distribution.png': {
    title: 'Drug Degree Distribution',
    desc: 'Histogram of drug node degrees in PrimeKG. High-degree drugs appear in many edges and can dominate predictions.'
  },
  'degree_vs_score.png': {
    title: 'Degree vs. Predicted Score',
    desc: 'Scatter plot of drug degree against mean predicted score with Spearman ρ annotation. Lower correlation = less bias.'
  },
  'roc_pr_curves.png': {
    title: 'ROC & Precision-Recall Curves',
    desc: 'Receiver operating characteristic and precision-recall curves on the held-out test set.'
  },
  'degree_stratified_metrics.png': {
    title: 'Degree-Stratified AUC',
    desc: 'Test AUC broken down by drug degree bucket (low/medium/high). Uniform performance indicates less bias.'
  },
  'topk_diversity.png': {
    title: 'Top-K Diversity (Jaccard)',
    desc: 'Jaccard similarity distribution of top-K drug lists across random disease pairs. Lower = more diverse predictions.'
  },
};

// ─── Main Component ───────────────────────────────────

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('discover');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState('');
  const [openViewers, setOpenViewers] = useState<Record<string, boolean>>({});
  const [activeTarget, setActiveTarget] = useState<TargetProtein | null>(null);

  // Metrics & plots
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [plotFiles, setPlotFiles] = useState<string[]>([]);
  const [metricsLoading, setMetricsLoading] = useState(false);

  // Fetch metrics on mount
  useEffect(() => {
    const fetchData = async () => {
      setMetricsLoading(true);
      try {
        const [metricsRes, plotsRes] = await Promise.all([
          axios.get(`${API_URL}/metrics`),
          axios.get(`${API_URL}/plots-list`),
        ]);
        setMetrics(metricsRes.data);
        setPlotFiles(plotsRes.data.plots || []);
      } catch {
        // Silently fail — user will see empty state
      } finally {
        setMetricsLoading(false);
      }
    };
    fetchData();
  }, []);

  const toggleViewer = (id: string) => {
    setOpenViewers(prev => ({ ...prev, [id]: !prev[id] }));
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
      const response = await axios.post(`${API_URL}/predict`, {
        disease: query,
        top_k: 12
      });
      setResults(response.data);
      if (response.data.targets?.length > 0) {
        setActiveTarget(response.data.targets[0]);
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

  const getScoreColor = (score: number) => {
    if (score > 0.8) return 'var(--success)';
    if (score > 0.6) return 'var(--primary)';
    if (score > 0.4) return 'var(--warning)';
    return 'var(--danger)';
  };

  const fmt = (n: number, decimals = 4) => {
    if (n == null || isNaN(n)) return '—';
    return n.toFixed(decimals);
  };

  const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

  // ─── Render Tabs ──────────────────────────────────

  return (
    <div className="app-container">
      <motion.header
        className="header"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <h1>Drug Repurposing AI</h1>
        <p>Graph Neural Network pipeline for discovering novel drug-disease associations from PrimeKG.</p>
      </motion.header>

      {/* Tab Navigation */}
      <motion.nav
        className="tab-nav"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <button
          className={`tab-btn ${activeTab === 'discover' ? 'active' : ''}`}
          onClick={() => setActiveTab('discover')}
        >
          <Search size={16} /> Drug Discovery
        </button>
        <button
          className={`tab-btn ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          <BarChart3 size={16} /> Model Performance
        </button>
        <button
          className={`tab-btn ${activeTab === 'bias' ? 'active' : ''}`}
          onClick={() => setActiveTab('bias')}
        >
          <Shield size={16} /> Bias Analysis
        </button>
      </motion.nav>

      <AnimatePresence mode="wait">
        {/* ─── TAB 1: Drug Discovery ──────────────────── */}
        {activeTab === 'discover' && (
          <motion.div
            key="discover"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
          >
            <div className="search-container">
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
            </div>

            {loading && (
              <div className="loading-container">
                <div className="loader"></div>
                <div className="loading-text">Querying the knowledge graph...</div>
              </div>
            )}

            {error && (
              <div className="error-message">
                <ShieldAlert style={{ marginBottom: '1rem', display: 'inline-block' }} size={32} />
                <p>{error}</p>
              </div>
            )}

            {!loading && !error && !results && (
              <div className="empty-state">
                <Database size={64} />
                <h2>Ready for Analysis</h2>
                <p>Enter a disease above to find potential repurposed drug candidates.</p>
              </div>
            )}

            {results && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="results-header" style={{ marginBottom: '2rem' }}>
                  <h2 className="results-title">
                    Results for <span className="results-disease-name">{results.disease}</span>
                  </h2>
                </div>

                {/* Target Proteins */}
                {results.targets?.length > 0 && activeTarget && (
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
                            color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', transition: 'all 0.2s'
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

                {/* Drug Cards */}
                <div className="cards-grid">
                  {results.predictions.map((pred, idx) => (
                    <motion.div
                      key={pred.drug_name}
                      className="drug-card"
                      initial={{ y: 50, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ duration: 0.5, delay: idx * 0.08 }}
                    >
                      <div className={`rank-badge rank-${idx + 1}`}>{idx + 1}</div>
                      <h3 className="drug-name">{pred.drug_name}</h3>

                      <div className="metrics-container">
                        {/* GNN Score */}
                        <div className="metric">
                          <div className="metric-header">
                            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                              <Activity size={14} /> GNN Score
                            </span>
                            <span className="metric-value">{pred.gnn_score.toFixed(4)}</span>
                          </div>
                          <div className="progress-bg">
                            <div
                              className="progress-fill"
                              style={{ width: `${pred.gnn_score * 100}%`, background: getScoreColor(pred.gnn_score) }}
                            ></div>
                          </div>
                        </div>

                        {/* Degree Info */}
                        <div className="metric">
                          <div className="metric-header">
                            <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                              <GitBranch size={14} /> Graph Degree
                            </span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                              <span className="metric-value">{pred.degree}</span>
                              <span className={`degree-badge ${pred.degree_bucket}`}>{pred.degree_bucket}</span>
                            </span>
                          </div>
                        </div>

                        {/* 3D Structure Viewer Toggle */}
                        {pred.ligand_url && (
                          <div style={{ marginTop: '0.5rem', paddingTop: '0.8rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                            <button
                              onClick={() => toggleViewer(pred.drug_name)}
                              style={{
                                width: '100%', padding: '0.5rem',
                                background: 'rgba(59, 130, 246, 0.1)', color: '#60a5fa',
                                border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '6px',
                                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                gap: '0.5rem', transition: 'all 0.2s', fontSize: '0.9rem'
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
                                  <MolecularViewer url={pred.ligand_url} format="sdf" title={pred.drug_name} />
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
          </motion.div>
        )}

        {/* ─── TAB 2: Model Performance ──────────────── */}
        {activeTab === 'performance' && (
          <motion.div
            key="performance"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
          >
            {metricsLoading && (
              <div className="loading-container">
                <div className="loader"></div>
                <div className="loading-text">Loading metrics...</div>
              </div>
            )}

            {!metricsLoading && !metrics && (
              <div className="empty-state">
                <AlertTriangle size={64} />
                <h2>No Metrics Available</h2>
                <p>Start the backend server to load training metrics.</p>
              </div>
            )}

            {metrics && (
              <>
                {/* Key Stats */}
                <div className="stats-grid">
                  <motion.div className="stat-card" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
                    <div className="stat-value">{fmt(metrics.test.test_auc, 3)}</div>
                    <div className="stat-label">Test ROC-AUC</div>
                    <div className="stat-sublabel">Link prediction accuracy</div>
                  </motion.div>
                  <motion.div className="stat-card" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.15 }}>
                    <div className="stat-value">{fmt(metrics.test.test_ap, 3)}</div>
                    <div className="stat-label">Test Avg Precision</div>
                    <div className="stat-sublabel">Ranking quality</div>
                  </motion.div>
                  <motion.div className="stat-card" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
                    <div className="stat-value">{metrics.best.best_epoch}</div>
                    <div className="stat-label">Best Epoch</div>
                    <div className="stat-sublabel">Early stopped at {metrics.config.epochs}</div>
                  </motion.div>
                  <motion.div className="stat-card" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.25 }}>
                    <div className="stat-value">{fmt(metrics.best.best_val_mrr, 4)}</div>
                    <div className="stat-label">Best Val MRR</div>
                    <div className="stat-sublabel">Mean reciprocal rank</div>
                  </motion.div>
                  <motion.div className="stat-card" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}>
                    <div className="stat-value">{pct(metrics.test['test_hits@10'])}</div>
                    <div className="stat-label">Test Hits@10</div>
                    <div className="stat-sublabel">True drug in top 10</div>
                  </motion.div>
                </div>

                {/* Training Config Table */}
                <h3 className="section-title"><Layers size={20} /> Training Configuration</h3>
                <p className="section-subtitle">Architecture and hyperparameters used for this training run.</p>
                <table className="config-table">
                  <thead>
                    <tr><th>Parameter</th><th>Value</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>Hidden Dimension</td><td>{metrics.config.hidden_dim}</td></tr>
                    <tr><td>Embedding Dimension</td><td>{metrics.config.embedding_dim}</td></tr>
                    <tr><td>GCN Layers</td><td>3 (1 input + 2 residual)</td></tr>
                    <tr><td>Dropout Rate</td><td>{metrics.config.dropout}</td></tr>
                    <tr><td>Learning Rate</td><td>{metrics.config.lr}</td></tr>
                    <tr><td>Weight Decay</td><td>{metrics.config.weight_decay}</td></tr>
                    <tr><td>Negative Ratio</td><td>{metrics.config.negative_ratio}:1</td></tr>
                    <tr><td>Neg Sampling</td><td>Inv-sqrt degree weighted</td></tr>
                    <tr><td>Early Stopping Patience</td><td>{metrics.config.patience} epochs</td></tr>
                    <tr><td>LR Scheduler</td><td>ReduceLROnPlateau (factor={metrics.config.lr_scheduler_factor})</td></tr>
                    <tr><td>Degree Correlation λ</td><td>{metrics.config.degree_corr_lambda}</td></tr>
                    <tr><td>Train Positives</td><td>{metrics.splits.train_pos.toLocaleString()}</td></tr>
                    <tr><td>Train Negatives</td><td>{metrics.splits.train_neg.toLocaleString()}</td></tr>
                    <tr><td>Val / Test Positives</td><td>{metrics.splits.val_pos.toLocaleString()} / {metrics.splits.test_pos.toLocaleString()}</td></tr>
                  </tbody>
                </table>

                {/* Diagnostic Plots */}
                <h3 className="section-title"><BarChart3 size={20} /> Diagnostic Plots</h3>
                <p className="section-subtitle">Visualizations generated during training and evaluation.</p>

                <div className="plots-grid">
                  {plotFiles.map((filename, idx) => {
                    const info = PLOT_INFO[filename] || { title: filename.replace('.png', '').replace(/_/g, ' '), desc: '' };
                    return (
                      <motion.div
                        key={filename}
                        className="plot-card"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.1 }}
                      >
                        <div className="plot-card-header">
                          <div className="plot-card-title">{info.title}</div>
                          <div className="plot-card-desc">{info.desc}</div>
                        </div>
                        <img
                          src={`${API_URL}/plots/${filename}`}
                          alt={info.title}
                          loading="lazy"
                        />
                      </motion.div>
                    );
                  })}
                </div>
              </>
            )}
          </motion.div>
        )}

        {/* ─── TAB 3: Bias Analysis ──────────────────── */}
        {activeTab === 'bias' && (
          <motion.div
            key="bias"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
          >
            {!metrics && (
              <div className="empty-state">
                <AlertTriangle size={64} />
                <h2>No Bias Data Available</h2>
                <p>Start the backend server to load analysis results.</p>
              </div>
            )}

            {metrics && (
              <>
                {/* Bias Metrics Cards */}
                <h3 className="section-title"><Shield size={20} /> Hub Bias Diagnostics</h3>
                <p className="section-subtitle">
                  Hub bias occurs when the model predicts high-degree (highly connected) drugs for every disease,
                  regardless of the actual biological relationship. These metrics quantify the bias level.
                </p>

                <div className="bias-grid">
                  <motion.div className="bias-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                    <div className="bias-card-title"><TrendingUp size={16} /> Spearman Correlation (ρ)</div>
                    <div className="bias-value" style={{ color: metrics.bias.spearman_rho > 0.5 ? '#f87171' : metrics.bias.spearman_rho > 0.3 ? '#fbbf24' : '#34d399' }}>
                      {fmt(metrics.bias.spearman_rho, 3)}
                    </div>
                    <div className="bias-desc">
                      Correlation between drug degree and predicted score. Ideal: &lt; 0.3.
                      {metrics.bias.spearman_rho > 0.5
                        ? ' ⚠️ Significant hub bias remains — high-degree drugs dominate predictions.'
                        : ' ✅ Hub bias is within acceptable range.'}
                    </div>
                  </motion.div>

                  <motion.div className="bias-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
                    <div className="bias-card-title"><Target size={16} /> Top-1 Mode Fraction</div>
                    <div className="bias-value" style={{ color: metrics.bias.top1_mode_fraction > 0.3 ? '#f87171' : '#34d399' }}>
                      {pct(metrics.bias.top1_mode_fraction)}
                    </div>
                    <div className="bias-desc">
                      Fraction of diseases where the same drug ({metrics.bias.top1_mode_drug_name}) ranks #1. Ideal: &lt; 10%.
                    </div>
                  </motion.div>

                  <motion.div className="bias-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                    <div className="bias-card-title"><Layers size={16} /> Mean Jaccard Similarity</div>
                    <div className="bias-value" style={{ color: metrics.bias.mean_jaccard > 0.4 ? '#f87171' : metrics.bias.mean_jaccard > 0.25 ? '#fbbf24' : '#34d399' }}>
                      {fmt(metrics.bias.mean_jaccard, 3)}
                    </div>
                    <div className="bias-desc">
                      Average top-K overlap across disease pairs. Lower means more diverse predictions. Ideal: &lt; 0.25.
                    </div>
                  </motion.div>

                  <motion.div className="bias-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
                    <div className="bias-card-title"><Zap size={16} /> P90 Jaccard</div>
                    <div className="bias-value" style={{ color: metrics.bias.p90_jaccard > 0.6 ? '#f87171' : '#fbbf24' }}>
                      {fmt(metrics.bias.p90_jaccard, 3)}
                    </div>
                    <div className="bias-desc">
                      90th percentile Jaccard across {metrics.bias.pairs_compared} disease pairs. High values indicate near-identical top-K lists.
                    </div>
                  </motion.div>
                </div>

                {/* Degree-Stratified AUC */}
                <h3 className="section-title"><BarChart3 size={20} /> Degree-Stratified Performance</h3>
                <p className="section-subtitle">
                  Test AUC broken down by drug degree bucket. If the model only works for high-degree drugs,
                  the "low" bucket will have significantly worse AUC.
                </p>

                <div style={{ background: 'var(--surface)', border: '1px solid var(--card-border)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2.5rem' }}>
                  <div className="strat-bars">
                    {(['low', 'medium', 'high'] as const).map(bucket => {
                      const data = metrics.degree_stratified[bucket];
                      const auc = data?.auc ?? 0;
                      const label = isNaN(auc) ? '—' : fmt(auc, 3);
                      return (
                        <div className="strat-row" key={bucket}>
                          <div className="strat-label">{bucket}</div>
                          <div className="strat-bar-bg">
                            <motion.div
                              className={`strat-bar-fill ${bucket}`}
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.max(isNaN(auc) ? 0 : auc * 100, 2)}%` }}
                              transition={{ duration: 1, delay: 0.3 }}
                            >
                              {label}
                            </motion.div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Degree thresholds: q33 = {metrics.degree_stratified.thresholds.q33}, q66 = {metrics.degree_stratified.thresholds.q66}
                  </div>
                </div>

                {/* What Was Done */}
                <h3 className="section-title"><CheckCircle size={20} /> Anti-Bias Countermeasures Applied</h3>
                <p className="section-subtitle">Changes implemented in this training pipeline to mitigate hub bias.</p>

                <ul className="changelog">
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>Test edge leakage fix:</strong> Removed val/test drug-disease edges from the training adjacency matrix. The model no longer sees test answers during training.</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>Inverse-degree negative sampling:</strong> Negatives are now sampled proportional to 1/√(degree), so high-degree drugs receive proportionally more negatives.</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>{metrics.config.negative_ratio}:1 negative ratio:</strong> Increased from 1:1 to better reflect the true sparsity of drug-disease space (&gt;99% negative).</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>Residual GCN layers:</strong> 3-layer GCN with skip connections and LayerNorm to prevent over-smoothing of embeddings.</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>Degree-aware scoring:</strong> The link predictor receives log(degree) as an explicit feature, allowing the model to discount degree influence.</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>Degree correlation loss (λ={metrics.config.degree_corr_lambda}):</strong> A regularization penalty that penalizes correlation between predicted scores and node degree.</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>Early stopping (patience={metrics.config.patience}):</strong> Monitored on validation MRR to prevent overfitting to degree patterns.</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>LR scheduling:</strong> ReduceLROnPlateau with factor={metrics.config.lr_scheduler_factor} and gradient clipping at {metrics.config.grad_clip_norm}.</span>
                  </li>
                  <li>
                    <span className="change-icon">✓</span>
                    <span><strong>Comprehensive evaluation:</strong> ROC-AUC, AP, Hits@K, MRR, degree-stratified AUC, Spearman ρ, and Jaccard diversity — not just AUC alone.</span>
                  </li>
                </ul>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
