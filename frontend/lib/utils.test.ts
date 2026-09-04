import { describe, expect, it } from "vitest";
import { decisionStyle, formatInr } from "./lib/utils";

describe("formatters", () => {
  it("formats INR", () => {
    expect(formatInr(1200)).toContain("1,200");
  });
  it("colors block decisions", () => {
    expect(decisionStyle("BLOCK")).toContain("flare");
  });
});
