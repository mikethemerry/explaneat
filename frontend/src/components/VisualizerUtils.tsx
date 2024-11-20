import { CSSProperties } from "react";

export type EdgeStyle = Pick<
  CSSProperties,
  "stroke" | "strokeWidth" | "strokeDasharray"
>;
export const getEdgeStyle = (weight: number): EdgeStyle => {
  const isNegative = weight < 0;
  const absWeight = Math.abs(weight);

  let strokeWidth = 1;
  if (absWeight >= 0.3 && absWeight <= 1) {
    // Linearly interpolate width between 1 and 3 for weights 0.3 to 1
    strokeWidth = 1 + (absWeight - 0.3) * (2 / 0.7);
  } else if (absWeight > 1) {
    strokeWidth = 3;
  }

  return {
    stroke: isNegative ? "#ff0000" : "#0000ff",
    strokeWidth: strokeWidth,
    strokeDasharray:
      absWeight < 0.1 ? "1,3" : absWeight < 0.3 ? "5,5" : undefined,
  };
};

export const getAnimatedBool = (weight: number): boolean => {
  return false;
  const absWeight = Math.abs(weight);
  return absWeight >= 0.1 && absWeight <= 0.3;
};
