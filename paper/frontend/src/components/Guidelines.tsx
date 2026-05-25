'use client';

import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface HeadChoice {
  value: string;
  label: string;
  help?: string;
}

interface HeadDef {
  head: string;
  title: string;
  description: string;
  choices: HeadChoice[];
}

interface Props {
  guidelines: HeadDef[];
}

export default function Guidelines({ guidelines }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-slate-200 rounded-lg bg-white overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-left font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
      >
        <span>Annotation Guidelines</span>
        {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </button>
      {open && (
        <div className="px-4 pb-4 border-t border-slate-100">
          <div className="space-y-5 mt-4">
            {guidelines.map((g) => (
              <div key={g.head}>
                <div className="font-bold text-slate-800 mb-1">
                  {g.title} <span className="text-slate-400 font-normal">— {g.description}</span>
                </div>
                <ul className="space-y-1 text-sm text-slate-600 ml-4">
                  {g.choices.map((c) => (
                    <li key={c.value}>
                      <span className="font-semibold text-slate-700">{c.label}</span>
                      {c.help && <span className="text-slate-500">: {c.help}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
