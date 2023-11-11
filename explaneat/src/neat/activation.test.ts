import { expect, test } from 'vitest'

import { ActivationFunction } from "./activation";

test('ActivationFunction.Sigmoid(0) should return 0.5', () => {
    expect(ActivationFunction.Sigmoid(0)).toBe(0.5);
});
test('ActivationFunction.Sigmoid(1) should return 0.7310585786300049', () => {
    expect(ActivationFunction.Sigmoid(1)).toBe(0.7310585786300049);
});
test('ActivationFunction.Sigmoid(-1) should return 0.2689414213699951', () => {
    expect(ActivationFunction.Sigmoid(-1)).toBe(0.2689414213699951);
});
test('ActivationFunction.RELU(0) should return 0', () => {
    expect(ActivationFunction.RELU(0)).toBe(0);
});
test('ActivationFunction.Relu(1) should return 1', () => {
    expect(ActivationFunction.RELU(1)).toBe(1);
});
test('ActivationFunction.Relu(-1) should return 0', () => {
    expect(ActivationFunction.RELU(-1)).toBe(0);
});
test('ActivationFunction.Identity(0) should return 0', () => {
    expect(ActivationFunction.Identity(0)).toBe(0);
});
test('ActivationFunction.Identity(1) should return 1', () => {
    expect(ActivationFunction.Identity(1)).toBe(1);
});
test('ActivationFunction.Identity(-1) should return -1', () => {
    expect(ActivationFunction.Identity(-1)).toBe(-1);
});
