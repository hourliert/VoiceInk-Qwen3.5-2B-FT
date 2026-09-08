import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";

const sections = [
  {
    label: "Workspace",
    links: [
      ["/", "Overview"],
      ["/review/recent", "Recent"],
      ["/review/queue?review=required", "Review queue"],
      ["/review/history", "History"],
    ],
  },
  {
    label: "Data",
    links: [
      ["/data/cohorts", "Cohorts"],
      ["/data/releases", "Releases"],
    ],
  },
  {
    label: "Models & runs",
    links: [
      ["/models", "Models"],
      ["/runs/training", "Training"],
      ["/runs/evaluations", "Evaluations"],
    ],
  },
  {
    label: "Operations",
    links: [
      ["/system", "System"],
      ["/system/docs", "Documentation"],
    ],
  },
];

export function Layout({children}: {children: ReactNode}) {
  const location = useLocation();
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink to="/" className="brand">
          <span className="brand-mark">V</span>
          <span>
            <strong>VoiceInk</strong>
            <small>Control Plane</small>
          </span>
        </NavLink>
        <nav aria-label="Primary navigation">
          {sections.map((section) => (
            <section className="nav-section" key={section.label}>
              <p>{section.label}</p>
              {section.links.map(([href, label]) => {
                const pathname = href.split("?")[0];
                const active = pathname === "/"
                  ? location.pathname === "/"
                  : location.pathname.startsWith(pathname);
                return (
                  <NavLink
                    to={href}
                    className={active ? "nav-link active" : "nav-link"}
                    key={href}
                  >
                    {label}
                  </NavLink>
                );
              })}
            </section>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="status-dot" />
          LAN · port 8003
        </div>
      </aside>
      <main className="main-content">{children}</main>
    </div>
  );
}
