import type { ReactNode } from "react";

type SectionCardProps = {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  className?: string;
  testId?: string;
};

export function SectionCard({
  title,
  eyebrow,
  children,
  className = "",
  testId,
}: SectionCardProps) {
  return (
    <section
      data-testid={testId}
      className={`raisa-card p-5 ${className}`}
    >
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3 border-b border-surface-3-light pb-3">
        <div>
          {eyebrow ? (
            <p className="text-[0.68rem] uppercase tracking-[0.28em] text-its-blue">
              {eyebrow}
            </p>
          ) : null}
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.03em] text-text-primary">
            {title}
          </h2>
        </div>
      </div>
      {children}
    </section>
  );
}
