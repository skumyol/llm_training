'use client';

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
  return (
    <div className="border border-slate-200 rounded-lg bg-white overflow-hidden">
      <div className="px-4 py-3 font-semibold text-slate-700 border-b border-slate-100 bg-slate-50">
        Annotation Guidelines
      </div>
      <div className="px-4 pb-4">
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
    </div>
  );
}
