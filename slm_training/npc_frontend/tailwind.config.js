/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          950: '#070b14',
          900: '#0d1117',
          800: '#111827',
          700: '#1a2235',
        },
        npc:    { DEFAULT: '#a78bfa', dark: '#1e1040' },
        player: { DEFAULT: '#34d399', dark: '#0d2b22' },
        amber:  { DEFAULT: '#f59e0b' },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
