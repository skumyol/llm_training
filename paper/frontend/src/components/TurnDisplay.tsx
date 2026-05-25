'use client';

interface HistoryMessage {
  speaker: string;
  text: string;
}

interface TurnData {
  scenario: string;
  episode: string;
  turn_number: string | number;
  scene: string;
  history: HistoryMessage[];
  player: string;
  npc: string;
}

interface Props {
  turn: TurnData | null;
}

export default function TurnDisplay({ turn }: Props) {
  if (!turn) {
    return (
      <div className="text-center text-slate-400 py-12">
        Enter your name and click <strong>Begin Audit</strong> to start.
      </div>
    );
  }

  const sceneLines = turn.scene
    ? turn.scene.split('\n').map((l) => l.trim()).filter(Boolean)
    : [];

  return (
    <div className="space-y-4">
      {/* Scenario badge */}
      <div className="flex items-center text-sm font-semibold text-slate-600 mb-2">
        <span className="bg-slate-200 px-3 py-1 rounded-full">{turn.scenario}</span>
        <span className="mx-2 text-slate-300">·</span>
        <span>Episode {turn.episode} · Turn {turn.turn_number}</span>
      </div>

      {/* Scene card */}
      {sceneLines.length > 0 && (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-sm space-y-1">
          {sceneLines.map((line, i) => {
            if (line.includes(':')) {
              const [key, val] = line.split(':', 2);
              return (
                <div key={i} className="flex gap-2">
                  <span className="text-slate-500 font-semibold uppercase text-xs tracking-wide">{key.trim()}</span>
                  <span className="text-slate-300">·</span>
                  <span className="text-slate-700 font-medium">{val.trim()}</span>
                </div>
              );
            }
            return <div key={i} className="text-slate-700">{line}</div>;
          })}
        </div>
      )}

      {/* Dialogue history */}
      {turn.history && turn.history.length > 0 && (
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Dialogue History</div>
          <div className="space-y-2">
            {turn.history.map((msg, i) => (
              <div key={i} className="flex items-start gap-2">
                {msg.speaker === 'Player' && (
                  <>
                    <span className="w-12 shrink-0 font-bold text-blue-700 text-xs pt-1">Player</span>
                    <span className="bg-blue-50 border-l-3 border-blue-400 px-3 py-1.5 rounded-r-lg text-blue-800 text-sm leading-relaxed flex-1">
                      {msg.text}
                    </span>
                  </>
                )}
                {msg.speaker === 'NPC' && (
                  <>
                    <span className="w-12 shrink-0 font-bold text-red-700 text-xs pt-1">NPC</span>
                    <span className="bg-red-50 border-l-3 border-red-400 px-3 py-1.5 rounded-r-lg text-red-800 text-sm leading-relaxed flex-1">
                      {msg.text}
                    </span>
                  </>
                )}
                {msg.speaker === 'System' && (
                  <span className="text-slate-500 text-sm pl-14">{msg.text}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Current turn */}
      <div className="space-y-3 pt-2">
        <div className="flex items-start gap-2">
          <span className="w-14 shrink-0 font-extrabold text-blue-800 text-sm pt-2">Player</span>
          <span className="bg-blue-50 border border-blue-300 px-4 py-3 rounded-lg text-blue-900 text-base leading-relaxed flex-1">
            {turn.player}
          </span>
        </div>
        <div className="flex items-start gap-2">
          <span className="w-14 shrink-0 font-extrabold text-red-800 text-sm pt-2">NPC</span>
          <span className="bg-red-50 border border-red-300 px-4 py-3 rounded-lg text-red-900 text-base leading-relaxed flex-1">
            {turn.npc}
          </span>
        </div>
      </div>
    </div>
  );
}
