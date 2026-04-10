import { useState, useEffect, useCallback } from 'react'
import { Cpu, Wifi, WifiOff, ChevronDown } from 'lucide-react'
import NPCPanel from './components/NPCPanel'
import ChatWindow from './components/ChatWindow'
import StatePanel from './components/StatePanel'
import AddNPCModal from './components/AddNPCModal'
import { api } from './api'

let msgCounter = 0
const nextId = () => ++msgCounter

export default function App() {
  const [health, setHealth]           = useState(null)
  const [npcs, setNpcs]               = useState([])
  const [activeNpc, setActiveNpc]     = useState(null)
  const [messages, setMessages]       = useState({})   // {npc_id: [msg, ...]}
  const [currentState, setCurrentState] = useState(null)
  const [memories, setMemories]       = useState([])
  const [evalData, setEvalData]       = useState(null)
  const [models, setModels]           = useState([])
  const [catalog, setCatalog]         = useState(null)
  const [showAddModal, setShowAddModal] = useState(false)
  const [showModels, setShowModels]   = useState(false)

  // ── Bootstrap ─────────────────────────────────────────────────────────────
  useEffect(() => {
    async function load() {
      try {
        const [h, n, e, m, c] = await Promise.allSettled([
          api.health(), api.listNpcs(), api.evalSummary(), api.models(), api.catalog(),
        ])
        if (h.status === 'fulfilled') setHealth(h.value.data)
        if (n.status === 'fulfilled') setNpcs(n.value.data.npcs)
        if (e.status === 'fulfilled') setEvalData(e.value.data)
        if (m.status === 'fulfilled') setModels(m.value.data.models)
        if (c.status === 'fulfilled') setCatalog(c.value.data)
      } catch {}
    }
    load()
    const iv = setInterval(async () => {
      try { const r = await api.health(); setHealth(r.data) } catch {}
    }, 10_000)
    return () => clearInterval(iv)
  }, [])

  // ── NPC helpers ───────────────────────────────────────────────────────────
  function handleSelect(npc_id) {
    setActiveNpc(npc_id)
    setCurrentState(null)
    setMemories([])
  }

  function handleAdded(newNpcs) {
    setNpcs(prev => {
      const ids = new Set(prev.map(n => n.npc_id))
      const merged = [...prev]
      for (const n of newNpcs) {
        if (!ids.has(n.npc_id)) merged.push(n)
      }
      return merged
    })
    if (!activeNpc && newNpcs.length > 0) setActiveNpc(newNpcs[0].npc_id)
  }

  function handleRemove(npc_id) {
    setNpcs(prev => prev.filter(n => n.npc_id !== npc_id))
    if (activeNpc === npc_id) {
      const remaining = npcs.filter(n => n.npc_id !== npc_id)
      setActiveNpc(remaining[0]?.npc_id ?? null)
    }
  }

  async function handleReset(npc_id) {
    setMessages(prev => ({ ...prev, [npc_id]: [] }))
    if (activeNpc === npc_id) { setCurrentState(null); setMemories([]) }
    setNpcs(prev => prev.map(n => n.npc_id === npc_id ? { ...n, turn_count: 0 } : n))
  }

  async function handleSelectModel(stage, model_id) {
    try {
      await api.selectModel(stage, model_id)
      const res = await api.catalog()
      setCatalog(res.data)
    } catch (err) {
      console.error('Model select failed:', err)
    }
  }

  // ── Chat ──────────────────────────────────────────────────────────────────
  const handleSend = useCallback(async (text) => {
    if (!activeNpc) return

    const playerMsg = {
      id: nextId(), role: 'player', content: text,
      ts: Date.now(), elapsed_ms: null,
    }
    setMessages(prev => ({
      ...prev,
      [activeNpc]: [...(prev[activeNpc] ?? []), playerMsg],
    }))

    const res = await api.chat(activeNpc, text)
    const { response, elapsed_ms, state, memories: mems } = res.data

    const npcMsg = {
      id: nextId(), role: 'npc', content: response,
      ts: Date.now(), elapsed_ms,
    }
    setMessages(prev => ({
      ...prev,
      [activeNpc]: [...(prev[activeNpc] ?? []), npcMsg],
    }))

    if (state && !state.error) setCurrentState(state)
    if (mems) setMemories(mems)

    setNpcs(prev => prev.map(n =>
      n.npc_id === activeNpc ? { ...n, turn_count: (n.turn_count ?? 0) + 2 } : n
    ))
  }, [activeNpc])

  // ── Derived ───────────────────────────────────────────────────────────────
  const activeProfile = npcs.find(n => n.npc_id === activeNpc)?.profile_text ?? ''
  const serviceReady  = health?.service_ready ?? false
  const connected     = health !== null

  const trainedRuns = models.filter(m => m.has_model)

  return (
    <div className="flex flex-col h-screen bg-surface-950">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-2.5 border-b border-slate-800 bg-surface-900 flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-amber-400 tracking-tight">⚔ NPC Chat Studio</span>
          <span className="text-xs text-slate-600 hidden sm:block">
            personality · affect · episodic memory
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Model selector */}
          <div className="relative">
            <button
              onClick={() => setShowModels(v => !v)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-xs text-slate-300 transition-colors"
            >
              <Cpu size={12} className="text-amber-400" />
              {trainedRuns.length > 0
                ? `${trainedRuns.length} model${trainedRuns.length > 1 ? 's' : ''}`
                : 'No models yet'}
              <ChevronDown size={11} className={`transition-transform ${showModels ? 'rotate-180' : ''}`} />
            </button>

            {showModels && (
              <div className="absolute right-0 top-full mt-1 w-72 bg-surface-800 border border-slate-700 rounded-xl shadow-2xl z-40 overflow-hidden">
                <div className="px-3 py-2 border-b border-slate-700">
                  <p className="text-xs font-semibold text-slate-400">Trained Models</p>
                </div>
                {models.length === 0 && (
                  <p className="text-xs text-slate-600 px-3 py-3">No trained models yet.</p>
                )}
                {models.map((m, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2 hover:bg-slate-700/50 transition-colors">
                    <div>
                      <p className="text-xs font-medium text-slate-300">{m.run_id}</p>
                      <p className="text-xs text-slate-600">{m.task} · {m.arch}</p>
                    </div>
                    <div className="text-right">
                      {m.best?.val_mse != null && (
                        <p className="text-xs text-slate-500">MSE {Number(m.best.val_mse).toFixed(4)}</p>
                      )}
                      {m.best?.val_ppl != null && (
                        <p className="text-xs text-slate-500">PPL {Number(m.best.val_ppl).toFixed(1)}</p>
                      )}
                      <span className={`text-xs ${m.has_model ? 'text-emerald-500' : 'text-slate-600'}`}>
                        {m.has_model ? '✓ saved' : 'no weights'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Connection status */}
          <div className={`flex items-center gap-1.5 text-xs ${
            !connected ? 'text-red-400' :
            serviceReady ? 'text-emerald-400' : 'text-yellow-400'
          }`}>
            {connected ? <Wifi size={13} /> : <WifiOff size={13} />}
            {!connected ? 'Offline' : serviceReady ? 'Ready' : 'Encoders only'}
          </div>
        </div>
      </header>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0" onClick={() => showModels && setShowModels(false)}>
        <NPCPanel
          npcs={npcs}
          activeNpc={activeNpc}
          onSelect={handleSelect}
          onAdd={() => setShowAddModal(true)}
          onRemove={handleRemove}
          onReset={handleReset}
        />

        <ChatWindow
          npcId={activeNpc}
          profile={activeProfile}
          messages={messages[activeNpc] ?? []}
          onSend={handleSend}
          onReset={handleReset}
          serviceReady={serviceReady}
        />

        <StatePanel
          state={currentState}
          memories={memories}
          evalData={evalData}
          serviceReady={serviceReady}
          catalog={catalog}
          onSelectModel={handleSelectModel}
        />
      </div>

      {/* ── Modals ──────────────────────────────────────────────────────── */}
      {showAddModal && (
        <AddNPCModal
          onClose={() => setShowAddModal(false)}
          onAdded={handleAdded}
        />
      )}
    </div>
  )
}
