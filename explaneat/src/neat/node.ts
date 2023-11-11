import { ActivationInterface } from "./activation";

enum NodeType {
    INPUT,
    OUTPUT,
    HIDDEN
}

class Node{
    id: string;
    type: NodeType;
    bias: number;
    response: number;
    activation: ActivationInterface;

    value: number;
    hasActivated: boolean;

    constructor(id: string, type: NodeType, bias: number, response: number, activation: ActivationInterface){
        this.id = id;
        this.type = type;
        this.bias = bias;
        this.response = response;
        this.activation = activation;

        this.value = 0;

        this.hasActivated = false;
    }

    activate(): void{
        this.response = this.activation(this.value);
        this.hasActivated = true;
    }
    getResponse(): number{
        if (!this.hasActivated){
            throw new Error("Node has not been activated yet");
        }
        return this.response;
    }
    getValue(): number{
        return this.value;
    }
    addValue(value: number): void{
        this.value += value;
    }
    reset(): void{
        this.value = 0;
        this.response = 0;
        this.hasActivated = false;
    }

}

export {Node, NodeType}