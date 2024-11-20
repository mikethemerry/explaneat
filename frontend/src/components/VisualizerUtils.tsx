import { CSSProperties } from "react";

export type EdgeStyle = Pick<
  CSSProperties,
  "stroke" | "strokeWidth" | "strokeDasharray"
>;
export const getEdgeStyle = (weight: number): EdgeStyle => {
  const isNegative = weight < 0;
  const absWeight = Math.abs(weight);

  return {
    stroke: isNegative ? "#ff0000" : "#0000ff",
    strokeWidth: absWeight > 0.5 ? 3 : 1,
    strokeDasharray: absWeight < 0.3 ? "5,5" : undefined,
  };
};

export const getAnimatedBool = (weight: number): boolean => {
  return weight < 0.3;
};
