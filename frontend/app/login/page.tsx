"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { api, TOKEN_KEY } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("admin@razorguard.local");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const data = await api<{ access_token: string }>("/api/v1/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      localStorage.setItem(TOKEN_KEY, data.access_token);
      toast.success("Signed in");
      router.push("/");
    } catch (err: any) {
      toast.error(err.message || "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen grid place-items-center px-4 py-8">
      <form onSubmit={submit} className="w-full max-w-md rounded-2xl border border-white/10 bg-ink-800 p-5 shadow-glow sm:p-8">
        <div className="mb-6">
          <div className="text-xs tracking-[0.2em] text-mint sm:tracking-[0.3em]">RAZORGUARD X</div>
          <h1 className="mt-2 text-xl font-semibold sm:text-2xl">Operator sign-in</h1>
          <p className="mt-2 text-sm text-slate-400">
            Independent student prototype. Local lab seed password is documented in the repo README. The
            production Render seed password is set only on the server and is not the lab default.
          </p>
        </div>
        <label className="text-xs text-slate-400">Email</label>
        <Input className="mt-1 mb-4" value={email} onChange={(e) => setEmail(e.target.value)} />
        <label className="text-xs text-slate-400">Password</label>
        <Input className="mt-1 mb-6" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
        <Button className="w-full" disabled={busy} type="submit">
          {busy ? "Checking…" : "Enter command floor"}
        </Button>
      </form>
    </div>
  );
}
