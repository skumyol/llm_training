import { useState } from 'react'
import { CheckCircle2, XCircle, ChevronRight, Terminal, Sparkles, RefreshCw } from 'lucide-react'
import { api } from '../api'

// ── Status badge ──────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  if (status === 'trained') {
    return (
      <span className="flex items-center gap-1 text-xs text-emerald-400 font-medium">
        <CheckCircle2 size={11} /> trained
      </span>
    )
  }
  if (status === 'training') {
    return (
      <span className="flex items-center gap-1 text-xs text-amber-400 font-medium animate-pulse">
        <RefreshCw size={11} className="animate-spin" /> training…
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1 text-xs text-slate-600 font-medium">
      <XCircle size={11} /> not trained
    </span>
  )
}

// ── Metric pill ───────────────────────────────────────────────────────────────
function MetricPill({ label, value }) {
  if (value == null) return null
  const num = typeof value === 'number' ? value : parseFloat(value)
  if (isNaN(num)) return null
  const formatted = num > 1000 ? `${(num / 1e6).toFixed(1)}M` : num.toFixed(num < 1 ? 4 : 1)
  return (
    <span className="text-xs bg-slate-800 border border-slate-700 rounded px-1.5 py-0.5 text-slate-400">
      {label} <span className="text-slate-300 font-mono">{formatted}</span>
    </span>
  )
}

// ── Train hint ────────────────────────────────────────────────────────────────
function TrainHint({ cmd }) {
  const [copied, setCopied] = useState(false)
  function copy() {
    navigator.clipboard.writeText(cmd)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div className="mt-2 bg-slate-900 border border-slate-700/60 rounded-lg px-3 py-2">
      <p className="text-xs text-slate-600 mb-1 flex items-center gap-1">
        <Terminal size={10} /> How to train
      </p>
      <div className="flex items-center gap-2">
        <code className="flex-1 text-xs text-slate-400 font-mono leading-relaxed break-all">{cmd}</code>
        <button
          onClick={copy}
          className="flex-shrink-0 text-xs text-slate-600 hover:text-amber-400 transition-colors px-1"
        >
          {copied ? '✓' : 'copy'}
        </button>
      </div>
    </div>
  )
}

// ── Model card ────────────────────────────────────────────────────────────────
function ModelCard({ model, stageId, isActive, onSelect, expanded, onToggleExpand }) {
  const trained   = model.status === 'trained'
  const selectable = trained

  return (
    <div
      className={`relative rounded-xl border transition-all duration-150 ${
        isActive
          ? 'border-amber-500/60 bg-amber-500/5 shadow-[0_0_0_1px_rgba(245,158,11,0.2)]'
          : trained
            ? 'border-slate-700 bg-slate-800/40 hover:border-slate-600 cursor-pointer'
            : 'border-slate-800 bg-slate-800/20 opacity-60'
      }`}
      onClick={() => !isActive && selectable && onSelect(stageId, model.id)}
    >
      {/* Active glow indicator */}
      {isActive && (
        <div className="absolute -top-px left-4 right-4 h-px bg-gradient-to-r from-transparent via-amber-500/60 to-transparent" />
      )}

      <div className="px-3 py-2.5">
        {/* Header row */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`text-xs font-semibold ${
                isActive ? 'text-amber-300' : trained ? 'text-slate-200' : 'text-slate-500'
              }`}>
                {model.name}
              </span>
              {isActive && (
                <span className="text-xs px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 font-medium">
                  active
                </span>
              )}
              {model.conditioned && (
                <span className="text-xs px-1.5 py-0.5 rounded-full bg-violet-900/40 text-violet-400 border border-violet-700/40">
                  <Sparkles size={9} className="inline mr-0.5" />conditioned
                </span>
              )}
            </div>
            <StatusBadge status={model.status} />
          </div>

          <button
            onClick={e => { e.stopPropagation(); onToggleExpand(model.id) }}
            className="flex-shrink-0 text-slate-600 hover:text-slate-400 p-0.5 transition-colors"
          >
            <ChevronRight size={13} className={`transition-transform ${expanded ? 'rotate-90' : ''}`} />
          </button>
        </div>

        {/* Metrics */}
        {trained && Object.keys(model.metrics || {}).length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2">
            <MetricPill label="val_ppl"  value={model.metrics.val_ppl}   />
            <MetricPill label="val_mse"  value={model.metrics.val_mse}   />
            <MetricPill label="MSE"      value={model.metrics.val_mse}   />
            <MetricPill label="params"   value={model.metrics.num_params} />
          </div>
        )}

        {/* Expanded detail */}
        {expanded && (
          <div className="mt-2 pt-2 border-t border-slate-700/50">
            <p className="text-xs text-slate-500 leading-relaxed">{model.desc}</p>
            {model.artifact && (
              <p className="text-xs text-slate-700 font-mono mt-1 truncate" title={model.artifact}>
                {model.artifact.split('/').slice(-3).join('/')}
              </p>
            )}
            {!trained && model.train_cmd && (
              <TrainHint cmd={model.train_cmd} />
            )}
            {trained && !isActive && selectable && (
              <button
                onClick={e => { e.stopPropagation(); onSelect(stageId, model.id) }}
                className="mt-2 w-full py-1.5 text-xs font-medium bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-lg transition-colors"
              >
                Switch to this model
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Stage section ─────────────────────────────────────────────────────────────
function StageSection({ stage, onSelect, expandedCard, onToggleExpand }) {
  const trainedCount = stage.options.filter(o => o.status === 'trained').length
  const totalCount   = stage.options.length

  return (
    <div className="mb-5">
      {/* Stage header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-xs font-semibold text-slate-300">{stage.label}</h3>
          <p className="text-xs text-slate-600 mt-0.5">{stage.description}</p>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full border ${
          trainedCount === totalCount
            ? 'bg-emerald-900/30 text-emerald-500 border-emerald-800/40'
            : trainedCount > 0
              ? 'bg-amber-900/30 text-amber-500 border-amber-800/40'
              : 'bg-slate-800 text-slate-600 border-slate-700'
        }`}>
          {trainedCount}/{totalCount}
        </span>
      </div>

      {/* Output arrow */}
      <div className="flex items-center gap-2 mb-2">
        <div className="h-px flex-1 bg-slate-800" />
        <span className="text-xs text-slate-700 font-mono">{stage.output}</span>
        <div className="h-px flex-1 bg-slate-800" />
      </div>

      {/* Model cards */}
      <div className="space-y-2">
        {stage.options.map(model => (
          <ModelCard
            key={model.id}
            model={model}
            stageId={stage.id}
            isActive={model.id === stage.active}
            onSelect={onSelect}
            expanded={expandedCard === model.id}
            onToggleExpand={onToggleExpand}
          />
        ))}
      </div>

      {/* Connector arrow to next stage */}
      <div className="flex justify-center mt-3">
        <div className="flex flex-col items-center">
          <div className="w-px h-3 bg-slate-700" />
          <ChevronRight size={12} className="text-slate-700 rotate-90" />
        </div>
      </div>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────
export default function ModelPipeline({ catalog, onSelect }) {
  const [expandedCard, setExpandedCard] = useState(null)

  function toggleExpand(id) {
    setExpandedCard(prev => prev === id ? null : id)
  }

  if (!catalog) {
    return (
      <div className="flex-1 flex items-center justify-center p-6 text-center">
        <p className="text-xs text-slate-600">Loading pipeline catalog…</p>
      </div>
    )
  }

  const stages = catalog.stages ?? []

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      {/* Legend */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <span className="flex items-center gap-1 text-xs text-slate-600">
          <CheckCircle2 size={10} className="text-emerald-500" /> trained
        </span>
        <span className="flex items-center gap-1 text-xs text-slate-600">
          <XCircle size={10} className="text-slate-600" /> not trained
        </span>
        <span className="flex items-center gap-1 text-xs text-slate-600">
          <Sparkles size={10} className="text-violet-400" /> personality/affect conditioned
        </span>
      </div>

      {stages.map((stage, i) => (
        <StageSection
          key={stage.id}
          stage={stage}
          onSelect={onSelect}
          expandedCard={expandedCard}
          onToggleExpand={toggleExpand}
        />
      ))}

      {/* Bottom note */}
      <p className="text-xs text-slate-700 text-center mt-2">
        Click any trained model to make it active · expand cards for train commands
      </p>
    </div>
  )
}
