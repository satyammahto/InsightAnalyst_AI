import { SeverityRiskLevel } from '../types';

export function trimf(x: number, abc: [number, number, number]): number {
  const [a, b, c] = abc;
  if (x <= a || x >= c) return 0.0;
  if (a < x && x <= b) return b !== a ? (x - a) / (b - a) : 1.0;
  return c !== b ? (c - x) / (c - b) : 1.0;
}

export function trapmf(x: number, abcd: [number, number, number, number]): number {
  const [a, b, c, d] = abcd;
  if (x <= a || x >= d) return 0.0;
  if (a < x && x < b) return b !== a ? (x - a) / (b - a) : 1.0;
  if (b <= x && x <= c) return 1.0;
  return d !== c ? (d - x) / (d - c) : 1.0;
}

export const INPUT_MF_PARAMS: {
  low: [number, number, number, number];
  medium: [number, number, number];
  high: [number, number, number, number];
} = {
  low: [0.0, 0.0, 20.0, 45.0],
  medium: [20.0, 50.0, 80.0],
  high: [55.0, 80.0, 100.0, 100.0]
};

export const OUTPUT_MF_PARAMS: {
  low: [number, number, number, number];
  medium: [number, number, number];
  high: [number, number, number];
  critical: [number, number, number, number];
} = {
  low: [0.0, 0.0, 15.0, 35.0],
  medium: [20.0, 42.5, 62.0],
  high: [48.0, 68.0, 85.0],
  critical: [72.0, 88.0, 100.0, 100.0]
};

export class MamdaniFuzzyInferenceSystem {
  universe: number[];

  constructor() {
    this.universe = [];
    for (let i = 0; i <= 200; i++) {
      this.universe.push((i / 200) * 100.0);
    }
  }

  fuzzifyInput(val: number): { low: number; medium: number; high: number } {
    const v = Math.max(0, Math.min(100, val));
    return {
      low: trapmf(v, INPUT_MF_PARAMS.low),
      medium: trimf(v, INPUT_MF_PARAMS.medium),
      high: trapmf(v, INPUT_MF_PARAMS.high)
    };
  }

  evaluate(
    intensity: number,
    deviation: number,
    impact: number
  ): {
    defuzzifiedScore: number;
    severityLabel: SeverityRiskLevel;
    ruleActivations: Record<string, number>;
  } {
    const muI = this.fuzzifyInput(intensity);
    const muD = this.fuzzifyInput(deviation);
    const muB = this.fuzzifyInput(impact);

    // Mamdani Rule Base (14 explicit rules)
    const r1 = Math.min(muI.low, muD.low, muB.low);
    const r2 = Math.max(muI.medium, muD.medium);
    const r3 = Math.min(muI.high, muD.high);
    const r4 = Math.min(muI.high, muB.high);
    const r5 = Math.min(muD.high, muB.high);
    const r6 = Math.min(muI.medium, muD.low);
    const r7 = Math.min(muI.medium, muB.low);
    const r8 = Math.min(muI.medium, muD.medium, muB.medium);
    const r9 = Math.min(muI.high, muD.high, muB.high);
    const r10 = Math.min(muI.low, muD.high);
    const r11 = Math.min(muI.low, muB.high);
    const r12 = Math.min(muI.high, muD.low, muB.low);
    const r13 = Math.min(muI.medium, muD.high, muB.low);
    const r14 = Math.min(muI.medium, muD.low, muB.high);

    // Output Class Aggregations (Max T-Conorm)
    const actLow = Math.max(r1, r6, r7);
    const actMed = Math.max(r2, r8, r10, r11);
    const actHigh = Math.max(r3, r12, r13, r14);
    const actCrit = Math.max(r4, r5, r9);

    // Centroid Defuzzification over Universe [0..100]
    let numerator = 0.0;
    let denominator = 0.0;

    for (const y of this.universe) {
      const muOutLow = Math.min(actLow, trapmf(y, OUTPUT_MF_PARAMS.low));
      const muOutMed = Math.min(actMed, trimf(y, OUTPUT_MF_PARAMS.medium));
      const muOutHigh = Math.min(actHigh, trimf(y, OUTPUT_MF_PARAMS.high));
      const muOutCrit = Math.min(actCrit, trapmf(y, OUTPUT_MF_PARAMS.critical));

      const muAgg = Math.max(muOutLow, muOutMed, muOutHigh, muOutCrit);
      numerator += y * muAgg;
      denominator += muAgg;
    }

    let defuzzifiedScore = 50.0;
    if (denominator > 1e-6) {
      defuzzifiedScore = numerator / denominator;
    } else {
      // Fallback weighted interpolation if no rules strongly fired
      defuzzifiedScore = (intensity * 0.35 + deviation * 0.45 + impact * 0.2);
    }
    defuzzifiedScore = Math.max(0.0, Math.min(100.0, defuzzifiedScore));

    let severityLabel: SeverityRiskLevel = 'Medium';
    if (defuzzifiedScore >= 75.0) {
      severityLabel = 'Critical';
    } else if (defuzzifiedScore >= 55.0) {
      severityLabel = 'High';
    } else if (defuzzifiedScore >= 35.0) {
      severityLabel = 'Medium';
    } else {
      severityLabel = 'Low';
    }

    return {
      defuzzifiedScore,
      severityLabel,
      ruleActivations: {
        Low: actLow,
        Medium: actMed,
        High: actHigh,
        Critical: actCrit
      }
    };
  }
}

export function getFuzzyMembershipCurves() {
  const universe: number[] = [];
  for (let i = 0; i <= 100; i += 2) {
    universe.push(i);
  }

  return {
    universe,
    inputs: {
      low: universe.map(x => trapmf(x, INPUT_MF_PARAMS.low)),
      medium: universe.map(x => trimf(x, INPUT_MF_PARAMS.medium)),
      high: universe.map(x => trapmf(x, INPUT_MF_PARAMS.high))
    },
    outputs: {
      low: universe.map(y => trapmf(y, OUTPUT_MF_PARAMS.low)),
      medium: universe.map(y => trimf(y, OUTPUT_MF_PARAMS.medium)),
      high: universe.map(y => trimf(y, OUTPUT_MF_PARAMS.high)),
      critical: universe.map(y => trapmf(y, OUTPUT_MF_PARAMS.critical))
    }
  };
}
