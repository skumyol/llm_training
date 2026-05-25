import AuditInterface from '@/components/AuditInterface';

interface PageProps {
  searchParams: { PROLIFIC_PID?: string; STUDY_ID?: string; SESSION_ID?: string };
}

export default function Home({ searchParams }: PageProps) {
  return (
    <AuditInterface
      initialProlificPid={searchParams.PROLIFIC_PID || ''}
      initialStudyId={searchParams.STUDY_ID || ''}
      initialSessionId={searchParams.SESSION_ID || ''}
    />
  );
}
