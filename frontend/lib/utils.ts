import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function decisionStyle(decision?: string | null) {
  if (decision === "BLOCK") return "text-flare bg-flare/10 border-flare/30";
  if (decision === "REVIEW") return "text-ember bg-ember/10 border-ember/30";
  if (decision === "APPROVE") return "text-mint bg-mint/10 border-mint/30";
  return "text-slate-300 bg-white/5 border-white/10";
}

export function formatInr(n?: number | null) {
  if (n == null) return "—";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(n);
}

export function formatTime(iso?: string | null) {
  if (!iso) return "—";
  // API datetimes are UTC. A naive ISO string is treated as local by Date(),
  // which shifts IST display by -5:30 (17:47 UTC → 5:47 PM instead of 11:17 PM).
  const hasTz = /[zZ]|[+-]\d{2}:?\d{2}$/.test(iso);
  return new Date(hasTz ? iso : `${iso}Z`).toLocaleString();
}
