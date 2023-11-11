import { expect, test } from 'vitest'
import { parseGenome, parseNode, parseEdge } from "./utiliities"
import { genome_1 } from './sample_data';

test('parseNode works', ()=>{
    const testNode = "0 DefaultNodeGene(key=0, bias=-0.7166086393996857, response=1.0, activation=relu, aggregation=sum)";
    const parsed = parseNode(testNode);
    expect(parsed.key).toBe(0);
    expect(parsed.data.bias).toBe(-0.7166086393996857);
    expect(parsed.data.response).toBe(1.0);
    expect(parsed.data.activation).toBe("relu");
    expect(parsed.data.aggregation).toBe("sum");
});

test('parseEdge works', ()=>{
    const testEdge = "        DefaultConnectionGene(key=(-9, 0), weight=0.09309050191856541, enabled=False)"
    const parsed = parseEdge(testEdge);
    expect(parsed.id).toBe("e-9-0");
    expect(parsed.source).toBe("-9");
    expect(parsed.target).toBe("0");
    expect(parsed.weight).toBe(0.09309050191856541);
    expect(parsed.enabled).toBe(false);
});


test('parseGenome works', ()=>{
    console.log(genome_1);
    const parsed = parseGenome(genome_1);
    console.log("parsed", parsed);
    expect(parsed.nodes.length).toBe(7);
});
