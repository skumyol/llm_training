import { useState } from 'react'
import { X, User, Globe } from 'lucide-react'
import { api } from '../api'

export default function AddNPCModal({ onClose, onAdded }) {
  const [tab, setTab] = useState('manual') // 'manual' | 'world'
  const [npcId, setNpcId] = useState('')
  const [profile, setProfile] = useState('')
  const [worldPath, setWorldPath] = useState('../../data/world_contexts/oakhaven_siege.yaml')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (tab === 'manual') {
        if (!npcId.trim() || !profile.trim()) {
          setError('Both NPC ID and profile are required.')
          setLoading(false)
          return
        }
        const res = await api.registerNpc(npcId.trim(), profile.trim())
        onAdded([{ npc_id: npcId.trim(), profile_text: profile.trim(), turn_count: 0 }])
      } else {
        const res = await api.loadWorld(worldPath.trim())
        const loaded = res.data.loaded.map(n => ({
          npc_id: n.npc_id, profile_text: n.profile_text, turn_count: 0
        }))
        onAdded(loaded)
      }
      onClose()
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-[520px] bg-surface-800 border border-slate-700 rounded-xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <h2 className="text-sm font-semibold text-slate-200">Register NPC</h2>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-700">
          {[['manual', <User size={13} />, 'Manual'], ['world', <Globe size={13} />, 'Load World']].map(([id, icon, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`flex items-center gap-2 px-5 py-2.5 text-xs font-medium transition-colors border-b-2 ${
                tab === id
                  ? 'border-amber-500 text-amber-400'
                  : 'border-transparent text-slate-500 hover:text-slate-300'
              }`}
            >
              {icon} {label}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {tab === 'manual' ? (
            <>
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">NPC ID</label>
                <input
                  autoFocus
                  value={npcId}
                  onChange={e => setNpcId(e.target.value)}
                  placeholder="commander_vance"
                  className="w-full bg-surface-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-amber-500 transition-colors"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">Profile Text</label>
                <textarea
                  value={profile}
                  onChange={e => setProfile(e.target.value)}
                  rows={5}
                  placeholder="A stoic castle guard who values duty above all. Speaks bluntly and distrusts strangers. Knows the supply shortage but won't reveal it."
                  className="w-full bg-surface-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-amber-500 transition-colors resize-none"
                />
                <p className="text-xs text-slate-600 mt-1">
                  Personality is auto-encoded from this text.
                </p>
              </div>
            </>
          ) : (
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">World YAML Path</label>
              <input
                autoFocus
                value={worldPath}
                onChange={e => setWorldPath(e.target.value)}
                className="w-full bg-surface-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-amber-500 transition-colors font-mono"
              />
              <p className="text-xs text-slate-600 mt-1">
                Relative to scaffold root or absolute. Loads all NPCs from the world context.
              </p>
            </div>
          )}

          {error && (
            <p className="text-xs text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs text-slate-400 hover:text-slate-200 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-5 py-2 bg-amber-500 hover:bg-amber-400 text-black text-xs font-semibold rounded-lg transition-colors disabled:opacity-50"
            >
              {loading ? 'Registering...' : tab === 'manual' ? 'Register NPC' : 'Load World'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
