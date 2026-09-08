import type { ReactNode } from "react";

export function Page({
  title,
  eyebrow,
  actions,
  children,
}: {
  title: string;
  eyebrow?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="page">
      <header className="page-header">
        <div>
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          <h1>{title}</h1>
        </div>
        {actions && <div className="page-actions">{actions}</div>}
      </header>
      {children}
    </div>
  );
}

export function Card({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={"card " + className}>
      {title && <h2>{title}</h2>}
      {children}
    </section>
  );
}

export function ErrorState({error}: {error: unknown}) {
  return (
    <div className="state-panel error-panel">
      <strong>Could not load this view</strong>
      <p>{error instanceof Error ? error.message : String(error)}</p>
    </div>
  );
}

export function Loading({label = "Loading"}: {label?: string}) {
  return (
    <div className="state-panel loading-panel" role="status">
      <span className="spinner" />
      {label}…
    </div>
  );
}

export function Empty({children}: {children: ReactNode}) {
  return <div className="state-panel empty-panel">{children}</div>;
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "info";
}) {
  return <span className={"badge " + tone}>{children}</span>;
}

export function Progress({
  value,
  total,
}: {
  value: number;
  total: number;
}) {
  const percent = total ? Math.min(100, Math.round(value / total * 100)) : 0;
  return (
    <div className="progress-wrap">
      <div className="progress-meta">
        <span>{value.toLocaleString()} / {total.toLocaleString()}</span>
        <span>{percent}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-bar" style={{width: String(percent) + "%"}} />
      </div>
    </div>
  );
}

type DiffPart = {kind: "same" | "add" | "remove"; text: string};

function wordDiff(base: string, target: string): DiffPart[] {
  const a = base.split(/(\s+)/).filter(Boolean);
  const b = target.split(/(\s+)/).filter(Boolean);
  if (a.length > 900 || b.length > 900) {
    return [{kind: "same", text: target}];
  }
  const width = b.length + 1;
  const rows = Array.from(
    {length: a.length + 1},
    () => new Uint16Array(width),
  );
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      rows[i][j] = a[i - 1] === b[j - 1]
        ? rows[i - 1][j - 1] + 1
        : Math.max(rows[i - 1][j], rows[i][j - 1]);
    }
  }
  const reversed: DiffPart[] = [];
  let i = a.length;
  let j = b.length;
  while (i || j) {
    if (i && j && a[i - 1] === b[j - 1]) {
      reversed.push({kind: "same", text: a[i - 1]});
      i -= 1;
      j -= 1;
    } else if (j && (!i || rows[i][j - 1] >= rows[i - 1][j])) {
      reversed.push({kind: "add", text: b[j - 1]});
      j -= 1;
    } else {
      reversed.push({kind: "remove", text: a[i - 1]});
      i -= 1;
    }
  }
  return reversed.reverse().reduce<DiffPart[]>((parts, part) => {
    const previous = parts[parts.length - 1];
    if (previous?.kind === part.kind) previous.text += part.text;
    else parts.push({...part});
    return parts;
  }, []);
}

export function DiffText({
  base,
  target,
  label,
}: {
  base: string;
  target: string;
  label: string;
}) {
  const parts = wordDiff(base, target);
  return (
    <div className="diff-block">
      <p className="diff-label">{label}</p>
      <div className="transcript diff-text">
        {parts.map((part, index) => (
          <span className={"diff-" + part.kind} key={String(index) + part.kind}>
            {part.text}
          </span>
        ))}
      </div>
    </div>
  );
}

export function ScoreGrid({
  production,
  proposal,
}: {
  production: Record<string, number>;
  proposal: Record<string, number>;
}) {
  const keys = Array.from(new Set([...Object.keys(production), ...Object.keys(proposal)]));
  return (
    <div className="score-grid">
      <div className="score-head">Dimension</div>
      <div className="score-head">Production</div>
      <div className="score-head">Luna</div>
      {keys.map((key) => (
        <div className="score-row" key={key}>
          <span>{key.replaceAll("_", " ")}</span>
          <strong>{production[key] ?? "–"}</strong>
          <strong>{proposal[key] ?? "–"}</strong>
        </div>
      ))}
    </div>
  );
}

export function formatDate(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}
