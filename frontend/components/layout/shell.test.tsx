/** @vitest-environment jsdom */
import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Shell } from "./shell";

vi.mock("next/navigation", () => ({
  usePathname: () => "/",
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));

vi.mock("next/link", () => ({
  default: ({ href, children, ...props }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

describe("Shell navigation", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    const store = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => {
        store.set(key, value);
      },
      removeItem: (key: string) => {
        store.delete(key);
      },
    });
    localStorage.setItem("rgx_token", "layout-test");
  });

  it("renders a mobile menu control and page content", () => {
    render(
      <Shell>
        <h1>Command floor</h1>
      </Shell>
    );
    expect(screen.getByLabelText("Open navigation")).toBeTruthy();
    expect(screen.getByText("Command floor")).toBeTruthy();
    expect(screen.getByText(/not a production fraud system/i)).toBeTruthy();
  });

  it("opens the drawer with the existing destinations", () => {
    render(
      <Shell>
        <div>body</div>
      </Shell>
    );
    fireEvent.click(screen.getAllByLabelText("Open navigation")[0]);
    expect(screen.getByLabelText("Close navigation overlay")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Command" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Live wire" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Entity graph" })).toBeTruthy();
    expect(screen.getByRole("button", { name: /sign out/i })).toBeTruthy();
  });
});
