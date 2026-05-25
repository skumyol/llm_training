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
  heads: HeadDef[];
  selections: Record<string, string>;
  placeholder: string;
  onChange: (head: string, value: string) => void;
  disabled?: boolean;
}

export default function LabelSelector({ heads, selections, placeholder, onChange, disabled }: Props) {
  const firstFour = heads.slice(0, 4);
  const rest = heads.slice(4);

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-slate-800">Select labels for each dimension</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {firstFour.map((h) => (
          <RadioGroup key={h.head} head={h} value={selections[h.head] || placeholder} placeholder={placeholder} onChange={onChange} disabled={disabled} />
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {rest.map((h) => (
          <RadioGroup key={h.head} head={h} value={selections[h.head] || placeholder} placeholder={placeholder} onChange={onChange} disabled={disabled} />
        ))}
      </div>
    </div>
  );
}

function RadioGroup({
  head,
  value,
  placeholder,
  onChange,
  disabled,
}: {
  head: HeadDef;
  value: string;
  placeholder: string;
  onChange: (head: string, value: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4">
      <div className="text-sm font-semibold text-slate-700 mb-1">{head.title}</div>
      <div className="text-xs text-slate-500 mb-3">{head.description}</div>
      <div className="flex flex-wrap gap-2">
        {head.choices.map((choice) => {
          const isSelected = value === choice.value;
          return (
            <button
              key={choice.value}
              type="button"
              disabled={disabled}
              onClick={() => onChange(head.head, choice.value)}
              className={[
                'px-3 py-1.5 rounded-md text-sm font-medium border transition-colors',
                isSelected
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100',
                disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
              ].join(' ')}
              title={choice.help || ''}
            >
              {choice.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
