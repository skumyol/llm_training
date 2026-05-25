'use client';

interface Props {
  elapsedThisTurn: number;
  timeRemaining: number;
  totalElapsed: number;
  minTime: number;
  testMode: boolean;
}

function fmtSeconds(total: number): string {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, '0')}m ${s.toString().padStart(2, '0')}s`;
}

export default function TimerDisplay({ elapsedThisTurn, timeRemaining, totalElapsed, minTime, testMode }: Props) {
  if (testMode) {
    return (
      <div className="text-sm font-medium text-slate-600">
        Turn time: {fmtSeconds(elapsedThisTurn)} | Session time: {fmtSeconds(totalElapsed)}
      </div>
    );
  }

  return (
    <div className="text-sm font-medium text-slate-600">
      Turn time: {elapsedThisTurn}s / {minTime}s min
      {timeRemaining > 0 && (
        <span className="text-amber-600 ml-2">Remaining: {timeRemaining}s</span>
      )}
      <span className="ml-2 text-slate-500">| Session time: {fmtSeconds(totalElapsed)}</span>
    </div>
  );
}
