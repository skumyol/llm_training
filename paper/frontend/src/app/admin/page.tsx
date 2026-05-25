'use client';

import { useState } from 'react';

const API_BASE = '/api';

export default function AdminPage() {
  const [teacherFile, setTeacherFile] = useState('../audit_input_clean.jsonl');
  const [auditDir, setAuditDir] = useState('../audit_results');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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

      <div className="bg-white border border-slate-200 rounded-lg p-4 mb-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-700">Run Evaluation</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1">Audit Directory</label>
            <input
              type="text"
              value={auditDir}
              onChange={(e) => setAuditDir(e.target.value)}
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1">Teacher File (optional)</label>
            <input
              type="text"
              value={teacherFile}
              onChange={(e) => setTeacherFile(e.target.value)}
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm"
            />
          </div>
        </div>
        <button
          onClick={runEvaluate}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-md disabled:opacity-50"
        >
          {loading ? 'Running...' : 'Evaluate All Auditors'}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-red-700 text-sm font-medium">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6">
          {/* QC Reports */}
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
                        }`}>
                          {r.overall_status}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-slate-600">
                        {r.flags?.length || 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pairwise Agreement */}
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
