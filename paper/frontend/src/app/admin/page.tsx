'use client';

import { useState } from 'react';

const API_BASE = '/api';

export default function AdminPage() {
  const [teacherFile, setTeacherFile] = useState('../audit_input_clean.jsonl');
  const [auditDir, setAuditDir] = useState('../audit_results');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Synthetic generation state
  const [synResult, setSynResult] = useState<any>(null);
  const [synLoading, setSynLoading] = useState(false);
  const [synHumanA, setSynHumanA] = useState('../audit_results/audit_654cfad67f990b0393b85132.jsonl');
  const [synHumanB, setSynHumanB] = useState('../audit_results/audit_67c87fc1b3ba111d0e1526a0.jsonl');
  const [synTarget, setSynTarget] = useState(0.40);
  const [synCount, setSynCount] = useState(3);
  const [synSeed, setSynSeed] = useState(42);

  // Complete & Generate state
  const [cgResult, setCgResult] = useState<any>(null);
  const [cgLoading, setCgLoading] = useState(false);
  const [cgTarget, setCgTarget] = useState(9);
  const [cgSeed, setCgSeed] = useState(42);

  const runCompleteAndGenerate = async () => {
    setCgLoading(true);
    setError('');
    setCgResult(null);
    try {
      const res = await fetch(`${API_BASE}/complete-and-generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_path: teacherFile,
          audit_dir: auditDir,
          target_total: cgTarget,
          seed: cgSeed,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      setCgResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setCgLoading(false);
    }
  };

  const runSynthetic = async () => {
    setSynLoading(true);
    setError('');
    setSynResult(null);
    try {
      const res = await fetch(`${API_BASE}/synthetic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_path: teacherFile,
          human_a: synHumanA,
          human_b: synHumanB,
          agreement_target: synTarget,
          count: synCount,
          seed: synSeed,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      setSynResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSynLoading(false);
    }
  };

  const runEvaluate = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audit_dir: auditDir, teacher_file: teacherFile || null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold text-slate-800 mb-6">Audit Admin Dashboard</h1>

      {/* Evaluation */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-700">Run Evaluation</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1">Audit Directory</label>
            <input type="text" value={auditDir} onChange={(e) => setAuditDir(e.target.value)}
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm" />
          </div>
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1">Teacher File (optional)</label>
            <input type="text" value={teacherFile} onChange={(e) => setTeacherFile(e.target.value)}
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm" />
          </div>
        </div>
        <button onClick={runEvaluate} disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-md disabled:opacity-50">
          {loading ? 'Running...' : 'Evaluate All Auditors'}
        </button>
      </div>

      {/* Synthetic Generation */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-700">Generate Synthetic Annotators</h2>
        <p className="text-xs text-slate-500">
          Creates synthetic annotations with a configurable statistical distribution between
          real human and teacher labels. Uses per-head confusion matrices from real annotators.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Human A File</label>
            <input type="text" value={synHumanA} onChange={(e) => setSynHumanA(e.target.value)}
              className="w-full border border-slate-300 rounded-md px-2 py-1.5 text-xs font-mono" />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Human B File</label>
            <input type="text" value={synHumanB} onChange={(e) => setSynHumanB(e.target.value)}
              className="w-full border border-slate-300 rounded-md px-2 py-1.5 text-xs font-mono" />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Agreement Target</label>
            <input type="number" step={0.01} min={0} max={1} value={synTarget}
              onChange={(e) => setSynTarget(Number(e.target.value))}
              className="w-24 border border-slate-300 rounded-md px-2 py-1.5 text-sm" />
            <span className="text-xs text-slate-400 ml-2">0.35=human-like, 0.50=mixed, 0.70=LLM-like</span>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Count / Seed</label>
            <div className="flex gap-2">
              <input type="number" min={1} max={20} value={synCount}
                onChange={(e) => setSynCount(Number(e.target.value))}
                className="w-20 border border-slate-300 rounded-md px-2 py-1.5 text-sm" placeholder="Count" />
              <input type="number" value={synSeed}
                onChange={(e) => setSynSeed(Number(e.target.value))}
                className="w-24 border border-slate-300 rounded-md px-2 py-1.5 text-sm" placeholder="Seed" />
            </div>
          </div>
        </div>
        <button onClick={runSynthetic} disabled={synLoading}
          className="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-md disabled:opacity-50">
          {synLoading ? 'Generating...' : 'Generate Synthetic Annotators'}
        </button>
      </div>

      {/* Complete & Generate */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-700">Complete & Generate</h2>
        <p className="text-xs text-slate-500">
          Automatically fills any partial human audits (&lt; 150 turns) with statistically coherent labels,
          then generates enough synthetic annotators to reach the target total.
        </p>
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Target Total</label>
            <input type="number" min={1} max={50} value={cgTarget}
              onChange={(e) => setCgTarget(Number(e.target.value))}
              className="w-24 border border-slate-300 rounded-md px-2 py-1.5 text-sm" />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Seed</label>
            <input type="number" value={cgSeed}
              onChange={(e) => setCgSeed(Number(e.target.value))}
              className="w-24 border border-slate-300 rounded-md px-2 py-1.5 text-sm" />
          </div>
        </div>
        <button onClick={runCompleteAndGenerate} disabled={cgLoading}
          className="bg-violet-600 hover:bg-violet-700 text-white font-semibold py-2 px-4 rounded-md disabled:opacity-50">
          {cgLoading ? 'Processing...' : 'Complete & Generate to Target'}
        </button>
      </div>

      {cgResult && (
        <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-3">Complete & Generate Results</h2>
          <div className="text-sm text-slate-700 mb-3">
            Full human: {cgResult.full_human_count} ·
            Filled human: {cgResult.filled_human_count} ·
            Existing synthetic: {cgResult.existing_synthetic_count} ·
            Generated synthetic: {cgResult.generated_synthetic_count} ·
            <strong> Total: {cgResult.actual_total} / {cgResult.target_total}</strong>
          </div>
          {cgResult.filled_audits.length > 0 && (
            <div className="mb-3">
              <div className="text-xs font-semibold text-slate-600 mb-1">Filled Partial Audits</div>
              {cgResult.filled_audits.map((a: any, i: number) => (
                <div key={i} className="text-xs text-slate-700 bg-slate-50 px-2 py-1 rounded mb-1">
                  {a.annotator}: {a.before} → {a.after} turns (+{a.filled} filled)
                </div>
              ))}
            </div>
          )}
          {cgResult.generated_synthetic.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-slate-600 mb-1">Generated Synthetic</div>
              {cgResult.generated_synthetic.map((a: any, i: number) => (
                <div key={i} className="text-xs text-slate-700 bg-slate-50 px-2 py-1 rounded mb-1">
                  {a.annotator}: {a.turns} turns → {a.file}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {synResult && (
        <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-3">Generated Annotators</h2>
          {synResult.synthetic_annotators.map((a: any, i: number) => (
            <div key={i} className="mb-3 last:mb-0 border border-slate-100 rounded-md p-3">
              <div className="font-bold text-slate-800 mb-1">{a.annotator}</div>
              <div className="text-xs text-slate-500 mb-2">
                {a.turns} turns · target={a.agreement_target} · {a.output_file}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-1 text-xs">
                {Object.entries(a.per_head_agreement).map(([head, acc]: [string, any]) => (
                  <div key={head} className="flex justify-between bg-slate-50 px-2 py-1 rounded">
                    <span className="text-slate-600">{head}</span>
                    <span className="font-mono font-semibold text-slate-800">{acc.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-red-700 text-sm font-medium">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-slate-700 mb-3">Quality Control Reports</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-slate-600">
                  <tr>
                    <th className="text-left px-3 py-2">Annotator</th>
                    <th className="text-left px-3 py-2">Turns</th>
                    <th className="text-left px-3 py-2">Median Time</th>
                    <th className="text-left px-3 py-2">Status</th>
                    <th className="text-left px-3 py-2">Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {result.qc_reports.map((r: any) => (
                    <tr key={r.name} className="border-t border-slate-100">
                      <td className="px-3 py-2 font-medium">{r.name}</td>
                      <td className="px-3 py-2">{r.n_turns}</td>
                      <td className="px-3 py-2">{r.median_turn_time_sec}s</td>
                      <td className="px-3 py-2">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${
                          r.overall_status === 'PASS' ? 'bg-green-100 text-green-800' :
                          r.overall_status === 'REVIEW' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>{r.overall_status}</span>
                      </td>
                      <td className="px-3 py-2 text-slate-600">{r.flags?.length || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {result.pairwise?.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-slate-700 mb-3">Pairwise Agreement</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 text-slate-600">
                    <tr>
                      <th className="text-left px-3 py-2">Pair</th>
                      <th className="text-left px-3 py-2">Avg Accuracy</th>
                      <th className="text-left px-3 py-2">Avg Kappa</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.pairwise.map((p: any, i: number) => (
                      <tr key={i} className="border-t border-slate-100">
                        <td className="px-3 py-2 font-medium">{p.a} vs {p.b}</td>
                        <td className="px-3 py-2">{p.avg_acc?.toFixed(3) || '--'}</td>
                        <td className="px-3 py-2">{p.avg_kappa?.toFixed(3) || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}