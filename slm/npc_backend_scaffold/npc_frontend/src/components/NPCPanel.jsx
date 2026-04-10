import { Plus, Trash2, RotateCcw } from 'lucide-react'
import { api } from '../api'

const AVATAR_COLORS = [
  'bg-violet-600', 'bg-amber-600', 'bg-emerald-600',
  'bg-rose-600',   'bg-sky-600',   'bg-orange-600',
]

function avatarColor(id) {
  let hash = 0
  for (const c of id) hash = (hash * 31 + c.charCodeAt(0)) & 0xffff
  return AVATAR_COLORS[hash % AVATAR_COLORS.length]
}

export default function NPCPanel({ npcs, activeNpc, onSelect, onAdd, onRemove, onReset }) {
  async function handleRemove(e, npc_id) {
    e.stopPropagation()
    try { await api.removeNpc(npc_id) } catch {}
    onRemove(npc_id)
  }

  async function handleReset(e, npc_id) {
    e.stopPropagation()
    try { await api.reset(npc_id) } catch {}
    onReset(npc_id)
  }

  return (
    <aside className="w-60 flex flex-col border-r border-slate-800 bg-surface-900">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">NPCs</span>
        <button
          onClick={onAdd}
          className="flex items-center gap-1 px-2 py-1 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 rounded-md text-xs font-medium transition-colors"
        >
          <Plus size={12} /> Add
        </button>
      </div>

      {/* NPC list */}
      <div className="flex-1 overflow-y-auto py-2">
        {npcs.length === 0 && (
          <div className="px-4 py-8 text-center">
            <p className="text-xs text-slate-600">No NPCs registered.</p>
            <p className="text-xs text-slate-700 mt-1">Click Add to get started.</p>
          </div>
        )}
        {npcs.map(npc => {
          const isActive = npc.npc_id === activeNpc
          return (
            <div
              key={npc.npc_id}
              onClick={() => onSelect(npc.npc_id)}
              className={`group flex items-start gap-3 mx-2 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                isActive
                  ? 'bg-amber-500/10 border border-amber-500/20'
                  : 'hover:bg-slate-800/60 border border-transparent'
              }`}
            >
              {/* Avatar */}
              <div className={`flex-shrink-0 w-8 h-8 rounded-full ${avatarColor(npc.npc_id)} flex items-center justify-center text-xs font-bold text-white`}>
                {npc.npc_id[0].toUpperCase()}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className={`text-xs font-semibold truncate ${isActive ? 'text-amber-400' : 'text-slate-300'}`}>
                  {npc.npc_id}
                </p>
                <p className="text-xs text-slate-600 truncate mt-0.5">
                  {npc.profile_text.slice(0, 40)}…
                </p>
                {npc.turn_count > 0 && (
                  <p className="text-xs text-slate-700 mt-0.5">{npc.turn_count} turns</p>
                )}
              </div>

              {/* Actions (show on hover) */}
              <div className="flex-shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={e => handleReset(e, npc.npc_id)}
                  className="p-1 text-slate-600 hover:text-amber-400 transition-colors"
                  title="Reset conversation"
                >
                  <RotateCcw size={11} />
                </button>
                <button
                  onClick={e => handleRemove(e, npc.npc_id)}
                  className="p-1 text-slate-600 hover:text-red-400 transition-colors"
                  title="Remove NPC"
                >
                  <Trash2 size={11} />
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </aside>
  )
}
