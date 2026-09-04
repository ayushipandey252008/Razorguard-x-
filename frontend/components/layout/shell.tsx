"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  Cpu,
  GitFork,
  LayoutDashboard,
  LogOut,
  Radar,
  ScrollText,
  ShieldAlert,
  Sparkles,
  LineChart,
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
  if (path === "/login") return <>{children}</>;

  if (typeof window !== "undefined" && !localStorage.getItem(TOKEN_KEY)) {
    router.replace("/login");
  }

  function logout() {
    localStorage.removeItem(TOKEN_KEY);
    router.push("/login");
  }

  return (
    <div className="min-h-screen grid grid-cols-[240px_1fr] bg-ink-950 text-slate-100">
      <aside className="border-r border-white/8 bg-ink-900/80 px-4 py-5 flex flex-col">
        <div className="flex items-center gap-3 px-2 mb-8">
          <div className="h-9 w-9 grid place-items-center rounded-lg bg-mint/15 text-mint font-mono text-lg font-bold border border-mint/30">
            X
          </div>
          <div>
            <div className="text-sm font-semibold tracking-[0.18em]">RAZORGUARD</div>
            <div className="text-[10px] uppercase tracking-[0.28em] text-mint/80">X · prototype</div>
          </div>
        </div>
        <nav className="space-y-1 flex-1">
          {NAV.map((item) => {
            const active = item.href === "/" ? path === "/" : path.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
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
          className="flex items-center gap-2 px-3 py-2 text-xs text-slate-500 hover:text-slate-200"
        >
          <LogOut size={14} /> Sign out
        </button>
      </aside>
      <div className="min-w-0">
        <div className="border-b border-ember/20 bg-ember/10 px-6 py-2 text-[11px] tracking-wide text-ember flex items-center gap-2">
          <ScrollText size={14} />
          Prototype / synthetic data — not a production fraud system, not affiliated with Razorpay.
        </div>
        <main className="p-6 lg:p-8">{children}</main>
      </div>
    </div>
  );
}
