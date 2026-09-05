"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  Cpu,
  GitFork,
  LayoutDashboard,
  LogOut,
  Menu,
  Radar,
  ScrollText,
  ShieldAlert,
  Sparkles,
  LineChart,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { TOKEN_KEY } from "@/lib/api";

const NAV = [
  { href: "/", label: "Command", icon: LayoutDashboard },
  { href: "/transactions", label: "Live wire", icon: Activity },
  { href: "/investigations", label: "Case files", icon: ShieldAlert },
  { href: "/fraud-network", label: "Entity graph", icon: GitFork },
  { href: "/analytics", label: "Telemetry", icon: Radar },
  { href: "/monitoring", label: "Monitoring", icon: LineChart },
  { href: "/simulation", label: "Range", icon: Sparkles },
  { href: "/scenario-eval", label: "Scenarios", icon: Sparkles },
  { href: "/system", label: "Model bay", icon: Cpu },
  { href: "/ieee-eval", label: "IEEE-CIS", icon: ScrollText },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
  }, [path]);

  useEffect(() => {
    if (!navOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNavOpen(false);
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [navOpen]);

  if (path === "/login") return <>{children}</>;

  if (typeof window !== "undefined" && !localStorage.getItem(TOKEN_KEY)) {
    router.replace("/login");
  }

  function logout() {
    localStorage.removeItem(TOKEN_KEY);
    router.push("/login");
  }

  return (
    <div className="min-h-screen bg-ink-950 text-slate-100 lg:grid lg:grid-cols-[240px_minmax(0,1fr)]">
      {navOpen ? (
        <button
          type="button"
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          aria-label="Close navigation overlay"
          onClick={() => setNavOpen(false)}
        />
      ) : null}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-[min(16.5rem,88vw)] flex-col border-r border-white/8 bg-ink-900 px-4 py-5 transition-transform duration-200 lg:static lg:z-auto lg:w-auto lg:translate-x-0 lg:bg-ink-900/80",
          navOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        <div className="mb-6 flex items-center justify-between gap-3 px-2 lg:mb-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-mint/30 bg-mint/15 font-mono text-lg font-bold text-mint">
              X
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold tracking-[0.18em]">RAZORGUARD</div>
              <div className="text-[10px] uppercase tracking-[0.28em] text-mint/80">X · prototype</div>
            </div>
          </div>
          <button
            type="button"
            className="grid min-h-11 min-w-11 place-items-center rounded-lg text-slate-400 hover:bg-white/5 hover:text-slate-100 lg:hidden"
            aria-label="Close navigation"
            onClick={() => setNavOpen(false)}
          >
            <X size={18} />
          </button>
        </div>
        <nav className="flex-1 space-y-1 overflow-y-auto">
          {NAV.map((item) => {
            const active = item.href === "/" ? path === "/" : path.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex min-h-11 items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                  active ? "bg-mint/10 text-mint" : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
                )}
              >
                <Icon size={16} />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <button
          onClick={logout}
          className="mt-3 flex min-h-11 items-center gap-2 px-3 py-2 text-xs text-slate-500 hover:text-slate-200"
        >
          <LogOut size={14} /> Sign out
        </button>
      </aside>
      <div className="flex min-w-0 flex-col">
        <div className="sticky top-0 z-30 border-b border-ember/20 bg-ember/10">
          <div className="flex items-start gap-2 px-3 py-2 text-[11px] leading-snug tracking-wide text-ember sm:px-6">
            <button
              type="button"
              className="mt-[-2px] grid min-h-11 min-w-11 shrink-0 place-items-center rounded-lg text-ember hover:bg-ember/15 lg:hidden"
              aria-label="Open navigation"
              aria-expanded={navOpen}
              onClick={() => setNavOpen(true)}
            >
              <Menu size={18} />
            </button>
            <div className="flex min-w-0 items-start gap-2 pt-2.5 lg:pt-0">
              <ScrollText size={14} className="mt-0.5 shrink-0" />
              <span className="min-w-0 text-pretty">
                Prototype / synthetic data — not a production fraud system, not affiliated with Razorpay.
              </span>
            </div>
          </div>
        </div>
        <main className="min-w-0 p-4 sm:p-6 lg:p-8">{children}</main>
      </div>
    </div>
  );
}
