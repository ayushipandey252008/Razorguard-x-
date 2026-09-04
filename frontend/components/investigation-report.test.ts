import { describe, expect, it } from "vitest";
import { ruleEvidenceState } from "./investigation-report";

describe("ruleEvidenceState", () => {
  it("does not treat an empty array as no rules fired", () => {
    expect(ruleEvidenceState({ triggered: [], note: null })).toBe("not_collected");
    expect(ruleEvidenceState({})).toBe("not_collected");
    expect(ruleEvidenceState(undefined)).toBe("not_collected");
  });

  it("uses the explicit tool note when zero rules were collected", () => {
    expect(
      ruleEvidenceState({ triggered: [], note: "No deterministic rules fired" })
    ).toBe("none_fired");
  });

  it("renders actual rules when triggered is populated", () => {
    expect(
      ruleEvidenceState({
        triggered: [{ rule_id: "HIGH_AMOUNT", rule_name: "Unusually high transaction amount" }],
        note: null,
      })
    ).toBe("rules");
  });
});
