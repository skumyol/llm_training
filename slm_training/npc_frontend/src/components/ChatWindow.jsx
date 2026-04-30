import { useEffect, useRef, useState } from 'react'
import { Send, RotateCcw } from 'lucide-react'
import { api } from '../api'

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 px-4 py-3">
      <div className="w-7 h-7 rounded-full bg-violet-700 flex items-center justify-center text-xs font-bold text-white flex-shrink-0">
        N
      </div>
      <div className="bg-surface-700 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1.5">
        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-violet-400 inline-block" />
        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-violet-400 inline-block" />
        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-violet-400 inline-block" />
      </div>
    </div>
  )
}

function MessageBubble({ msg, npcId }) {
  const isPlayer = msg.role === 'player'
  return (
    <div className={`flex items-start gap-3 px-4 py-2 ${isPlayer ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
        isPlayer ? 'bg-emerald-700 text-white' : 'bg-violet-700 text-white'
      }`}>
        {isPlayer ? 'P' : (npcId?.[0]?.toUpperCase() ?? 'N')}
      </div>

      {/* Bubble */}
      <div className={`max-w-[70%] ${isPlayer ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
        <div className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isPlayer
            ? 'bg-player-dark border border-emerald-800/50 text-emerald-100 rounded-tr-sm'
            : 'bg-npc-dark border border-violet-800/40 text-violet-100 rounded-tl-sm'
        }`}>
          {msg.content}
        </div>
        <div className="flex items-center gap-2 px-1">
          {msg.elapsed_ms && (
            <span className="text-xs text-slate-700">{msg.elapsed_ms}ms</span>
          )}
          <span className="text-xs text-slate-700">
            {new Date(msg.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
      </div>
    </div>
  )
}

export default function ChatWindow({ npcId, profile, messages, onSend, onReset, serviceReady }) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    setInput('')
    setError('')
    inputRef.current?.focus()
  }, [npcId])

  async function handleSend() {
    const text = input.trim()
    if (!text || loading || !npcId) return
    setInput('')
    setError('')
    setLoading(true)
    try {
      await onSend(text)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Request failed')
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  if (!npcId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
        <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4">
          <span className="text-2xl">⚔️</span>
        </div>
        <p className="text-sm text-slate-500">Select an NPC to start chatting</p>
        <p className="text-xs text-slate-700 mt-1">or register a new one from the sidebar</p>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-w-0">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800 bg-surface-900">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-slate-200">{npcId}</h2>
            {!serviceReady && (
              <span className="text-xs px-2 py-0.5 bg-yellow-900/40 text-yellow-500 border border-yellow-800/50 rounded-full">
                No model — train first
              </span>
            )}
          </div>
          <p className="text-xs text-slate-600 truncate mt-0.5">{profile}</p>
        </div>
        <button
          onClick={() => onReset(npcId)}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-500 hover:text-amber-400 hover:bg-amber-500/10 rounded-lg transition-colors"
        >
          <RotateCcw size={12} /> Reset
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-3">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <p className="text-xs text-slate-600">Say something to {npcId}…</p>
          </div>
        )}
        {messages.map(msg => (
          <MessageBubble key={msg.id} msg={msg} npcId={npcId} />
        ))}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 px-3 py-2 bg-red-900/20 border border-red-800/40 rounded-lg text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Input */}
      <div className="px-4 pb-4">
        <div className="flex items-end gap-2 bg-surface-800 border border-slate-700 rounded-xl p-2 focus-within:border-amber-500/50 transition-colors">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder={serviceReady ? `Message ${npcId}…` : 'Dialogue model not loaded yet…'}
            disabled={!serviceReady || loading}
            className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none leading-relaxed max-h-32 disabled:opacity-40"
            style={{ minHeight: '24px' }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading || !serviceReady}
            className="flex-shrink-0 w-8 h-8 bg-amber-500 hover:bg-amber-400 disabled:opacity-30 disabled:cursor-not-allowed text-black rounded-lg flex items-center justify-center transition-colors"
          >
            <Send size={14} />
          </button>
        </div>
        <p className="text-xs text-slate-700 mt-1.5 px-1">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  )
}
