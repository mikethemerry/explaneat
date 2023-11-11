interface ActivationInterface {
    (input: number): number;
}

const Sigmoid: ActivationFunction = (input:number) => {
    return 1 / (1 + Math.exp(-input));
}

const RELU: ActivationFunction = (input:number) => {
    return Math.max(0, input);
}

const Identity: ActivationFunction = (input:number) => {
    return input;
}

const ActivationFunction = {Sigmoid: Sigmoid, RELU: RELU, Identity: Identity};


export {ActivationFunction, ActivationInterface}