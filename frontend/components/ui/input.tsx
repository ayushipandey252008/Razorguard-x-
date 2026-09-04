import type { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "flex h-10 w-full rounded-md border border-white/10 bg-ink-900 px-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-mint/50 focus:outline-none focus:ring-1 focus:ring-mint/40",
        props.className
      )}
    />
  );
}
