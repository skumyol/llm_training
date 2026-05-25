'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import TurnDisplay from './TurnDisplay';
import LabelSelector from './LabelSelector';
import Guidelines from './Guidelines';
import TimerDisplay from './TimerDisplay';
import { ChevronDown, ChevronUp, AlertTriangle, CheckCircle } from 'lucide-react';

const API_BASE = '/api';
const MIN_TIME_PER_TURN = 30;

interface HeadDef {
  head: string;
  title: string;
  description: string;
  choices: { value: string; label: string; help?: string }[];
}

interface SessionState {
  annotator: string;
  index: number;
  total_turns: number;
  progress: string;
  is_done: boolean;
  test_mode: boolean;
  turn: any;
  previous_labels: Record<string, string>;
  previous_notes: string;
  time_remaining: number;
  can_submit: boolean;
  elapsed_this_turn: number;
  total_elapsed: number;
  annotated_count: number;
  completion_code: string;
}

interface AuditInterfaceProps {
  initialTestMode?: boolean;
  showDevControls?: boolean;
  initialProlificPid?: string;
  initialStudyId?: string;
  initialSessionId?: string;
}

export default function AuditInterface({ initialTestMode = false, showDevControls = false, initialProlificPid = '', initialStudyId = '', initialSessionId = '' }: AuditInterfaceProps) {
  const [annotatorName, setAnnotatorName] = useState(initialProlificPid);
  const [prolificPid, setProlificPid] = useState(initialProlificPid);
  const [studyId, setStudyId] = useState(initialStudyId);
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [testMode, setTestMode] = useState(initialTestMode);
  const [sampleSize, setSampleSize] = useState(150);
  const [session, setSession] = useState<SessionState | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [notes, setNotes] = useState('');
  const [warning, setWarning] = useState('');
  const [loading, setLoading] = useState(false);
  const [guidelines, setGuidelines] = useState<HeadDef[]>([]);
  const [infoOpen, setInfoOpen] = useState(true);
  const [infoText, setInfoText] = useState('');
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const handleBeginRef = useRef<() => void>(() => {});

  // Load guidelines and info on mount, auto-start if URL params present
  useEffect(() => {
    fetch(`${API_BASE}/guidelines`)
      .then((r) => r.json())
      .then((data) => setGuidelines(data.guidelines || []));
    fetch(`${API_BASE}/info`)
      .then((r) => r.json())
      .then((data) => setInfoText(data.info || ''));
    // Auto-start session if Prolific params are in the URL
    if (initialProlificPid) {
      setInfoOpen(false);
      handleBeginRef.current();
    }
  }, []);

  // Local timer tick (updates elapsed time every second)
  useEffect(() => {
    if (!isActive || !session || session.is_done) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    timerRef.current = setInterval(() => {
      setSession((prev) => {
        if (!prev) return prev;
        const newElapsed = prev.elapsed_this_turn + 1;
        const newTotal = prev.total_elapsed + 1;
        const minTime = prev.test_mode ? 0 : MIN_TIME_PER_TURN;
        const remaining = Math.max(0, minTime - newElapsed);
        return {
          ...prev,
          elapsed_this_turn: newElapsed,
          total_elapsed: newTotal,
          time_remaining: remaining,
          can_submit: remaining === 0,
        };
      });
    }, 1000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isActive, session?.is_done]);

  const handleBegin = async () => {
    setWarning('');
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          annotator_name: annotatorName,
          prolific_pid: prolificPid,
          study_id: studyId,
          session_id: sessionId,
          test_mode: testMode,
          sample_size: sampleSize,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setWarning(data.detail || 'Failed to start session.');
        return;
      }
      setSession(data);
      setIsActive(true);
      if (data.previous_labels) {
        setSelections(data.previous_labels);
        setNotes(data.previous_notes || '');
      } else {
        setSelections({});
        setNotes('');
      }
    } finally {
      setLoading(false);
    }
  };
  handleBeginRef.current = handleBegin;

  const handleEnd = async () => {
    if (!session) return;
    setWarning('');
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/session/${encodeURIComponent(session.annotator)}/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotator_name: session.annotator }),
      });
      const data = await res.json();
      if (!res.ok) {
        setWarning(data.detail || 'Failed to end session.');
        return;
      }
      setSession(data);
      setIsActive(false);
      setSelections({});
      setNotes('');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!session) return;

    const allSelected = guidelines.every((h) => selections[h.head] && selections[h.head] !== '-- select --');
    if (!session.test_mode && !allSelected) {
      setWarning('Please select a label for all 8 heads before submitting.');
      return;
    }
    if (!session.test_mode && session.time_remaining > 0) {
      setWarning(`Please wait ${session.time_remaining} more seconds before submitting this turn.`);
      return;
    }

    setWarning('');
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/session/${encodeURIComponent(session.annotator)}/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          annotator_name: session.annotator,
          labels: selections,
          notes: notes,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setWarning(data.detail || 'Submit failed.');
        return;
      }
      setSession(data);
      if (data.is_done) {
        setIsActive(false);
        setSelections({});
        setNotes('');
      } else {
        setSelections(data.previous_labels || {});
        setNotes(data.previous_notes || '');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleBack = async () => {
    if (!session) return;
    setWarning('');
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/session/${encodeURIComponent(session.annotator)}/back`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotator_name: session.annotator }),
      });
      const data = await res.json();
      if (!res.ok) {
        setWarning(data.detail || 'Cannot go back.');
        return;
      }
      setSession(data);
      setSelections(data.previous_labels || {});
      setNotes(data.previous_notes || '');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectionChange = useCallback((head: string, value: string) => {
    setSelections((prev) => ({ ...prev, [head]: value }));
  }, []);

  const isComplete = session?.is_done || false;
  const showAudit = isActive && !isComplete;

  return (
    <div className="flex gap-6 px-4 py-6 max-w-screen-2xl mx-auto">
      {/* Left sidebar — Guidelines always visible */}
      {guidelines.length > 0 && (
        <aside className="w-80 shrink-0">
          <div className="sticky top-6">
            <Guidelines guidelines={guidelines} />
          </div>
        </aside>
      )}

      <main className="flex-1 min-w-0">
        {/* Header */}
        <h1 className="text-2xl font-bold text-center text-slate-800 mb-6">NPC Social-State Human Audit</h1>

        {session?.test_mode && (
          <div className="bg-yellow-100 border-2 border-yellow-400 rounded-lg p-3 text-center font-bold text-yellow-900 mb-4">
            TEST MODE — Timer disabled, selections optional, for internal UX testing only
          </div>
        )}

        {/* Info accordion */}
        <div className="border border-slate-200 rounded-lg bg-white overflow-hidden mb-4">
          <button
            onClick={() => setInfoOpen(!infoOpen)}
            className="w-full flex items-center justify-between px-4 py-3 text-left font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
          >
            <span>Click here for experiment instructions and info</span>
            {infoOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
          {infoOpen && (
            <div className="px-4 pb-4 border-t border-slate-100">
              <div className="prose prose-slate max-w-none text-sm mt-3">
                <ReactMarkdown>{infoText}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>

        {/* Setup panel — hidden when Prolific URL params are present */}
        {!initialProlificPid && (
        <div className="bg-white border border-slate-200 rounded-lg p-4 mb-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-1">Prolific Participant ID</label>
              <input
                type="text"
                value={prolificPid}
                onChange={(e) => setProlificPid(e.target.value)}
                placeholder="Paste your Prolific ID here"
                disabled={isActive}
                className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
              />
              <p className="text-xs text-slate-500 mt-1">From your Prolific invitation URL. Used for your audit filename.</p>
            </div>
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-1">Annotator Name</label>
              <input
                type="text"
                value={annotatorName}
                onChange={(e) => setAnnotatorName(e.target.value)}
                placeholder="e.g. alice"
                disabled={isActive}
                className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
              />
              <p className="text-xs text-slate-500 mt-1">Your unique identifier for this audit session.</p>
            </div>
            <div className="flex items-end">
              {!isActive ? (
                <button
                  onClick={handleBegin}
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-4 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Starting...' : 'Begin Audit'}
                </button>
              ) : (
                <button
                  onClick={handleEnd}
                  disabled={loading}
                  className="w-full bg-slate-600 hover:bg-slate-700 text-white font-semibold py-2.5 px-4 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Ending...' : 'End Audit'}
                </button>
              )}
            </div>
          </div>

          {/* Dev controls (only shown on /test route) */}
          {showDevControls && (
            <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-slate-100">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={testMode}
                  onChange={(e) => setTestMode(e.target.checked)}
                  disabled={isActive}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 disabled:opacity-50"
                />
                <span className="text-sm font-medium text-slate-700">Test Mode</span>
                <span className="text-xs text-slate-500">(no timer, selections optional)</span>
              </label>
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-slate-700">Sample Size</label>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={sampleSize}
                  onChange={(e) => setSampleSize(Number(e.target.value))}
                  disabled={isActive}
                  className="w-20 border border-slate-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
                />
                <span className="text-xs text-slate-500">turns</span>
              </div>
            </div>
          )}
        </div>
        )}

        {/* Timer */}
        {session && (
          <div className="mb-4">
            <TimerDisplay
              elapsedThisTurn={session.elapsed_this_turn}
              timeRemaining={session.time_remaining}
              totalElapsed={session.total_elapsed}
              minTime={MIN_TIME_PER_TURN}
              testMode={session.test_mode}
            />
          </div>
        )}

        {/* Progress */}
        {session && (
          <div className="text-lg font-bold text-slate-800 mb-4">{session.progress}</div>
        )}

        {/* Warning */}
        {warning && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 flex items-start gap-2">
            <AlertTriangle className="text-red-600 shrink-0 mt-0.5" size={18} />
            <span className="text-red-700 font-medium text-sm">{warning}</span>
          </div>
        )}

        {/* Done message */}
        {isComplete && session && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6 text-center">
            <CheckCircle className="mx-auto text-green-600 mb-3" size={40} />
            <h2 className="text-xl font-bold text-green-800 mb-2">All {session.total_turns} turns annotated!</h2>
            <p className="text-slate-700 mb-4">Your audit has been saved.</p>
            <div className="inline-block bg-slate-100 rounded-lg px-4 py-3">
              <p className="text-sm text-slate-600 mb-1">Prolific Completion Code</p>
              <code className="text-lg font-bold text-slate-800 bg-white px-3 py-1 rounded border border-slate-200">
                {session.completion_code}
              </code>
            </div>
            <p className="text-xs text-slate-500 mt-3">
              Copy this code and paste it into Prolific to receive your payment.
            </p>
          </div>
        )}

        {/* Early end message */}
        {!isActive && session && !isComplete && (
          <div className="bg-orange-50 border border-orange-200 rounded-lg p-6 mb-6 text-center">
            <h2 className="text-xl font-bold text-orange-800 mb-2">Audit ended early</h2>
            <p className="text-slate-700">
              Annotated {session.annotated_count} / {session.total_turns} turns.
            </p>
            <p className="text-slate-500 text-sm mt-2">
              You must complete all turns to receive the completion code.
            </p>
          </div>
        )}

        {/* Turn display */}
        {showAudit && session && (
          <div className="bg-white border border-slate-200 rounded-lg p-5 mb-4">
            <TurnDisplay turn={session.turn} />
          </div>
        )}

        {/* Label selectors */}
        {showAudit && session && guidelines.length > 0 && (
          <div className="mb-4">
            <LabelSelector
              heads={guidelines}
              selections={selections}
              placeholder="-- select --"
              onChange={handleSelectionChange}
              disabled={loading}
            />
          </div>
        )}

        {/* Notes */}
        {showAudit && session && (
          <div className="mb-4">
            <label className="block text-sm font-semibold text-slate-700 mb-1">Notes (optional)</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Ambiguity, disagreements, anything unusual..."
              rows={2}
              disabled={loading}
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
            />
          </div>
        )}

        {/* Action buttons */}
        {showAudit && session && (
          <div className="flex gap-3">
            <button
              onClick={handleBack}
              disabled={loading || session.index === 0}
              className="bg-slate-200 hover:bg-slate-300 text-slate-800 font-semibold py-2.5 px-5 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous Turn
            </button>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-5 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Submitting...' : 'Submit & Next Turn'}
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
