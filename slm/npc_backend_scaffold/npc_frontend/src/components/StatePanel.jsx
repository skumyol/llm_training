import { useState } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip as ReTooltip,
} from 'recharts'
import { Brain, Zap, Database, GitBranch } from 'lucide-react'
import ModelPipeline from './ModelPipeline'

const OCEAN_LABELS = { openness: 'Open', conscientiousness: 'Consc', extraversion: 'Extra', agreeableness: 'Agree', neuroticism: 'Neuro' }
const VAD_META = {
  valence:   { label: 'Valence',   desc: 'mood',    lo: 'negative', hi: 'positive', color: '#34d399', bg: 'bg-emerald-500' },
  arousal:   { label: 'Arousal',   desc: 'energy',  lo: 'calm',     hi: 'excited',  color: '#60a5fa', bg: 'bg-blue-500'    },
  dominance: { label: 'Dominance', desc: 'control', lo: 'submissive',hi: 'dominant', color: '#a78bfa', bg: 'bg-violet-500' },
}

function Bar({ meta, value }) {
  const pct = Math.round((value ?? 0.5) * 100)
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium">{meta.label}</span>
        <span className="text-slate-500 font-mono">{(value ?? 0.5).toFixed(3)}</span>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500`}
          style={{ width: `${pct}%`, backgroundColor: meta.color }}
        />
      </div>
      <div className="flex justify-between text-xs text-slate-700">
        <span>{meta.lo}</span><span>{meta.hi}</span>
      </div>
    </div>
  )
}

function PersonalityRadar({ personality }) {
  if (!personality) return null
  const data = Object.entries(personality).map(([key, val]) => ({
    dim: OCEAN_LABELS[key] || key,
    value: Math.round(val * 100),
    fullMark: 100,
  }))
  return (
    <ResponsiveContainer width="100%" height={180}>
      <RadarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 16 }}>
        <PolarGrid stroke="#1f2937" />
        <PolarAngleAxis
          dataKey="dim"
          tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'JetBrains Mono' }}
        />
        <Radar
          dataKey="value"
          stroke="#a78bfa"
          fill="#a78bfa"
          fillOpacity={0.18}
          strokeWidth={1.5}
        />
        <ReTooltip
          contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }}
          formatter={(v) => [`${v}%`, '']}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}

function EvalBadge({ evalData }) {
  if (!evalData?.affect?.best) return null
  const { val_mse } = evalData.affect.best
  const { val_r2 } = evalData.affect.best?.metrics ?? {}
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-500 border border-slate-700">
        affect MSE {Number(val_mse).toFixed(4)}
      </span>
      {val_r2 != null && (
        <span className={`text-xs px-2 py-0.5 rounded-full border ${
          val_r2 > 0.3 ? 'bg-emerald-900/30 text-emerald-400 border-emerald-800/50'
                       : 'bg-slate-800 text-slate-500 border-slate-700'
        }`}>
          R² {Number(val_r2).toFixed(3)}
        </span>
      )}
    </div>
  )
}

export default function StatePanel({ state, memories, evalData, serviceReady, catalog, onSelectModel }) {
  const [tab, setTab] = useState('state')
  const personality = state?.personality
  const affect      = state?.affect

  const TABS = [
    { id: 'state',    label: 'State',    icon: <Brain size={12} /> },
    { id: 'pipeline', label: 'Pipeline', icon: <GitBranch size={12} /> },
  ]

  return (
    <aside className="w-80 flex flex-col border-l border-slate-800 bg-surface-900">
      {/* Tab header */}
      <div className="flex border-b border-slate-800 flex-shrink-0">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 text-xs font-medium transition-colors border-b-2 ${
              tab === t.id
                ? 'border-amber-500 text-amber-400'
                : 'border-transparent text-slate-500 hover:text-slate-300'
            }`}
          >
            {t.icon} {t.label}
            {t.id === 'pipeline' && catalog && (
              <span className="ml-1 text-xs px-1.5 py-0.5 rounded-full bg-slate-800 text-slate-500 border border-slate-700">
                {catalog.stages?.reduce((n, s) => n + s.options.filter(o => o.status === 'trained').length, 0)}/
                {catalog.stages?.reduce((n, s) => n + s.options.length, 0)}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── State tab ──────────────────────────────────────────────────── */}
      {tab === 'state' && (
        <div className="flex-1 overflow-y-auto">
          {/* Service status */}
          <div className="px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
            <span className="text-xs text-slate-500">Service</span>
            <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${
              serviceReady
                ? 'bg-emerald-900/30 text-emerald-400 border-emerald-800/50'
                : 'bg-yellow-900/30 text-yellow-500 border-yellow-800/50'
            }`}>
              {serviceReady ? 'Model ready' : 'Encoders only'}
            </span>
          </div>

          {!personality && (
            <div className="flex flex-col items-center justify-center py-12 text-center px-6">
              <Brain size={28} className="text-slate-700 mb-3" />
              <p className="text-xs text-slate-600">State appears after first NPC response.</p>
            </div>
          )}

          {/* Personality */}
          {personality && (
            <div className="px-4 pt-4 pb-3 border-b border-slate-800/60">
              <div className="flex items-center gap-2 mb-2">
                <Brain size={13} className="text-violet-400" />
                <span className="text-xs font-semibold text-slate-300">
                  Personality <span className="text-slate-600 font-normal">(OCEAN)</span>
                </span>
              </div>
              <PersonalityRadar personality={personality} />
              <div className="grid grid-cols-5 gap-1 mt-1">
                {Object.entries(personality).map(([k, v]) => (
                  <div key={k} className="text-center">
                    <div className="text-xs font-mono text-violet-400">{(v * 100).toFixed(0)}</div>
                    <div className="text-xs text-slate-700">{OCEAN_LABELS[k]?.slice(0, 1)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Affect */}
          {affect && (
            <div className="px-4 pt-4 pb-3 border-b border-slate-800/60 space-y-4">
              <div className="flex items-center gap-2 mb-1">
                <Zap size={13} className="text-amber-400" />
                <span className="text-xs font-semibold text-slate-300">
                  Affect <span className="text-slate-600 font-normal">(VAD)</span>
                </span>
              </div>
              {Object.entries(VAD_META).map(([key, meta]) => (
                <Bar key={key} meta={meta} value={affect[key]} />
              ))}
            </div>
          )}

          {/* Memories */}
          {memories && memories.length > 0 && (
            <div className="px-4 pt-4 pb-3 border-b border-slate-800/60">
              <div className="flex items-center gap-2 mb-2">
                <Database size={13} className="text-sky-400" />
                <span className="text-xs font-semibold text-slate-300">Retrieved Memories</span>
              </div>
              <div className="space-y-2">
                {memories.map((m, i) => (
                  <p key={i} className="text-xs text-slate-500 bg-slate-800/50 rounded-lg px-3 py-2 leading-relaxed">
                    {m}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Pipeline tab ───────────────────────────────────────────────── */}
      {tab === 'pipeline' && (
        <ModelPipeline catalog={catalog} onSelect={onSelectModel} />
      )}
    </aside>
  )
}
