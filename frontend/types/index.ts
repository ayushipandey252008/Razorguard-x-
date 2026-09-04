export type Decision = "APPROVE" | "REVIEW" | "BLOCK";

export type Transaction = {
  transaction_id: string;
  user_id: string;
  merchant_id: string;
  amount: number;
  currency: string;
  timestamp: string;
  device_id: string;
  ip_address: string;
  location: string;
  payment_method: string;
  merchant_category: string;
  account_age_days: number;
  failed_attempts: number;
  transaction_velocity: number;
  previous_transaction_count: number;
  previous_average_amount: number;
  current_device_known: boolean;
  current_location_known: boolean;
  payment_identifier: string;
  scenario_tag?: string | null;
  decision?: Decision | null;
  final_risk_score?: number | null;
};

export type TriggeredRule = {
  rule_id: string;
  rule_name: string;
  severity: string;
  score_contribution: number;
  explanation: string;
  evidence: Record<string, unknown>;
};

export type Risk = {
  transaction_id: string;
  ml_score: number;
  ml_probability: number;
  behavior_score: number;
  rule_score: number;
  graph_score: number;
  final_risk_score: number;
  decision: Decision;
  confidence: number;
  model_version: string;
  shap_top_features: { feature: string; contribution: number; value?: number | string }[];
  anomalies: { code: string; description: string; severity: string; contribution: number }[];
  graph_evidence: Record<string, any>;
  triggered_rules: TriggeredRule[];
  explanation: string;
  weights: Record<string, unknown>;
  probability_calibrated?: boolean;
  ml_probability_raw?: number | null;
};
