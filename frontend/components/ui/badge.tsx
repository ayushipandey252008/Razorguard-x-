import type { ReactNode } from "react";
import { cn, decisionStyle } from "@/lib/utils";

export function Badge({
  children,
  className,
  decision,
}: {
  children: ReactNode;
  className?: string;
  decision?: string | null;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-semibold tracking-wide",
        decision ? decisionStyle(decision) : "border-white/10 bg-white/5 text-slate-300",
        className
      )}
    >
      {children}
    </span>
  );
}
