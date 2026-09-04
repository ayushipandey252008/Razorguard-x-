"use client";

import { useEffect, useState } from "react";
import { wsUrl } from "@/lib/api";

export function useLiveEvents(onEvent: (e: any) => void) {
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let ws: WebSocket | null = null;
    let cancelled = false;
    try {
      ws = new WebSocket(wsUrl());
    } catch {
      return;
    }
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (m) => {
      try {
        onEvent(JSON.parse(m.data));
      } catch {
        /* ignore */
      }
    };
    return () => {
      cancelled = true;
      ws?.close();
    };
  }, [onEvent]);
  return connected;
}
