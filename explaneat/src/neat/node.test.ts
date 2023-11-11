import { expect, test } from 'vitest'

import { ActivationFunction } from "./activation";
import { Node } from "./node";


test("Node constructor works", ()=>{
    const node = new Node("node0", "default", 0, 1, ActivationFunction.RELU, true);
    expect(node.id).toBe("node0");
    expect(node.type).toBe("default");
    expect(node.bias).toBe(0);
    expect(node.response).toBe(1);
    expect(node.activation).toBe(ActivationFunction.RELU);
    expect(node.value).toBe(0);
    expect(node.hasActivated).toBe(false);
});

test("Node activate works", ()=>{
    const node = new Node("node0", "default", 0, 1, ActivationFunction.RELU, true);
    node.addValue(1);
    node.activate();
    expect(node.hasActivated).toBe(true);
    expect(node.getResponse()).toBe(1);
});