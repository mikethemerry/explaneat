import { expect, test } from 'vitest'
import { parseGenome, parseNode, parseEdge } from "./Utilities"
import { genome_1 } from './sample_data';

import { Node, Edge } from 'reactflow';


test('parseNode works', ()=>{
    const testNode = "0 DefaultNodeGene(key=0, bias=-0.7166086393996857, response=1.0, activation=relu, aggregation=sum)";
    const parsed = parseNode(testNode);
    expect(parsed.id).toBe("node0");
    expect(parsed.type).toBe("default");
    expect(parsed.data.label).toBe("Node 0");

});

test('parseEdge works', ()=>{
    const testEdge = "        DefaultConnectionGene(key=(-9, 0), weight=0.09309050191856541, enabled=False)"
    const parsed = parseEdge(testEdge);
    expect(parsed.id).toBe("edge-9-0");
    expect(parsed.source).toBe("node-9");
    expect(parsed.target).toBe("node0");
});


test('parseGenome works', ()=>{
    console.log(genome_1);
    const parsed = parseGenome(genome_1);
    // console.log("parsed", parsed);
    expect(parsed.nodes.length).toBe(16);


});
