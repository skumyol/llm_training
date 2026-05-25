import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'NPC Social-State Human Audit',
  description: 'Human validation audit for NPC dialogue generation research',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-800 min-h-screen">{children}</body>
    </html>
  );
}
