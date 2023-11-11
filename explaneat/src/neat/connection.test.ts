import { Node } from "./node";

class Connection{
    source: Node;
    target: Node;
    weight: number;
    enabled: boolean;

    constructor(source: Node, target: Node, weight: number, enabled: boolean){
        this.source = source;
        this.target = target;
        this.weight = weight;
        this.enabled = enabled;
    }

    getWeight(): number{
        return this.weight;
    }

    isEnabled(): boolean{
        return this.enabled;
    }

    setWeight(weight: number): void{
        this.weight = weight;
    }

    setEnabled(enabled: boolean): void{
        this.enabled = enabled;
    }

    reset(): void{
        this.weight = 0;
        this.enabled = false;
    }
}