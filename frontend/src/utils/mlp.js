/**
 * Custom Multi-Layer Perceptron (MLP) Neural Network
 * Implemented in vanilla JavaScript for instant in-browser training.
 * 
 * Architecture:
 * - Input Layer: 3 nodes [EAR, MAR, GazeYawDiff]
 * - Hidden Layer: N nodes (default 6) with Sigmoid activation
 * - Output Layer: 3 nodes [Alert (0), Drowsy (1), Distracted (2)] with Softmax activation
 */
export class SimpleMLP {
    constructor(inputDim = 3, hiddenDim = 6, outputDim = 3) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        
        // Initialize weights and biases randomly (Xavier/Glorot-like scale)
        const scale1 = Math.sqrt(2.0 / (inputDim + hiddenDim));
        this.w1 = Array.from({ length: inputDim }, () => 
            Array.from({ length: hiddenDim }, () => (Math.random() - 0.5) * 2 * scale1)
        );
        this.b1 = Array.from({ length: hiddenDim }, () => 0.0);
        
        const scale2 = Math.sqrt(2.0 / (hiddenDim + outputDim));
        this.w2 = Array.from({ length: hiddenDim }, () => 
            Array.from({ length: outputDim }, () => (Math.random() - 0.5) * 2 * scale2)
        );
        this.b2 = Array.from({ length: outputDim }, () => 0.0);
    }
    
    sigmoid(x) {
        return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
    }
    
    softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / (sum || 1));
    }
    
    // Forward propagation
    forward(inputs) {
        // inputs: array of size 3 [ear, mar, yaw_diff]
        const hidden = this.b1.map((bias, hIdx) => {
            let sum = bias;
            for (let i = 0; i < this.inputDim; i++) {
                sum += inputs[i] * this.w1[i][hIdx];
            }
            return this.sigmoid(sum);
        });
        
        const outputs = this.b2.map((bias, oIdx) => {
            let sum = bias;
            for (let h = 0; h < this.hiddenDim; h++) {
                sum += hidden[h] * this.w2[h][oIdx];
            }
            return sum;
        });
        
        const probs = this.softmax(outputs);
        return { hidden, probs };
    }
    
    // Perform a single epoch of training (Gradient Descent over all samples)
    trainStep(samples, lr = 0.05) {
        if (!samples || samples.length === 0) return { loss: 0, accuracy: 0 };
        
        let totalLoss = 0;
        let correct = 0;
        
        // Accumulators for gradients
        const dw1 = Array.from({ length: this.inputDim }, () => Array.from({ length: this.hiddenDim }, () => 0));
        const db1 = Array.from({ length: this.hiddenDim }, () => 0);
        const dw2 = Array.from({ length: this.hiddenDim }, () => Array.from({ length: this.outputDim }, () => 0));
        const db2 = Array.from({ length: this.outputDim }, () => 0);
        
        for (const sample of samples) {
            const { inputs, target } = sample;
            const { hidden, probs } = this.forward(inputs);
            
            // Cross-entropy loss (clip probabilities to avoid log(0))
            const probTarget = Math.max(1e-15, Math.min(1 - 1e-15, probs[target]));
            totalLoss -= Math.log(probTarget);
            
            // Check prediction
            const predClass = probs.indexOf(Math.max(...probs));
            if (predClass === target) {
                correct++;
            }
            
            // Output gradient (y_pred - y_true)
            const dOut = probs.map((p, idx) => idx === target ? p - 1 : p);
            
            // Accumulate hidden-to-output gradients
            for (let h = 0; h < this.hiddenDim; h++) {
                for (let o = 0; o < this.outputDim; o++) {
                    dw2[h][o] += dOut[o] * hidden[h];
                }
            }
            for (let o = 0; o < this.outputDim; o++) {
                db2[o] += dOut[o];
            }
            
            // Backpropagate to hidden layer
            const dHidden = this.b1.map((_, h) => {
                let sum = 0;
                for (let o = 0; o < this.outputDim; o++) {
                    sum += dOut[o] * this.w2[h][o];
                }
                // Derivative of sigmoid: hidden * (1 - hidden)
                return sum * hidden[h] * (1 - hidden[h]);
            });
            
            // Accumulate input-to-hidden gradients
            for (let i = 0; i < this.inputDim; i++) {
                for (let h = 0; h < this.hiddenDim; h++) {
                    dw1[i][h] += dHidden[h] * inputs[i];
                }
            }
            for (let h = 0; h < this.hiddenDim; h++) {
                db1[h] += dHidden[h];
            }
        }
        
        // Update weights/biases
        const N = samples.length;
        for (let i = 0; i < this.inputDim; i++) {
            for (let h = 0; h < this.hiddenDim; h++) {
                this.w1[i][h] -= (dw1[i][h] / N) * lr;
            }
        }
        for (let h = 0; h < this.hiddenDim; h++) {
            this.b1[h] -= (db1[h] / N) * lr;
        }
        for (let h = 0; h < this.hiddenDim; h++) {
            for (let o = 0; o < this.outputDim; o++) {
                this.w2[h][o] -= (dw2[h][o] / N) * lr;
            }
        }
        for (let o = 0; o < this.outputDim; o++) {
            this.b2[o] -= (db2[o] / N) * lr;
        }
        
        return {
            loss: totalLoss / N,
            accuracy: correct / N
        };
    }

    // Evaluate the model on test samples and calculate precision, recall, f1, and confusion matrix
    evaluate(samples) {
        if (!samples || samples.length === 0) {
            return {
                loss: 0,
                accuracy: 0,
                confusionMatrix: [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                metrics: [
                    { precision: 0, recall: 0, f1: 0 },
                    { precision: 0, recall: 0, f1: 0 },
                    { precision: 0, recall: 0, f1: 0 }
                ]
            };
        }
        
        let totalLoss = 0;
        let correct = 0;
        const confusionMatrix = [
            [0, 0, 0], // Actual Alert: Predicted [Alert, Drowsy, Distracted]
            [0, 0, 0], // Actual Drowsy: Predicted [Alert, Drowsy, Distracted]
            [0, 0, 0]  // Actual Distracted: Predicted [Alert, Drowsy, Distracted]
        ];
        
        for (const sample of samples) {
            const { inputs, target } = sample;
            const { probs } = this.forward(inputs);
            
            // Cross-entropy loss (clip probabilities to avoid log(0))
            const probTarget = Math.max(1e-15, Math.min(1 - 1e-15, probs[target]));
            totalLoss -= Math.log(probTarget);
            
            const predClass = probs.indexOf(Math.max(...probs));
            if (predClass === target) {
                correct++;
            }
            confusionMatrix[target][predClass]++;
        }
        
        // Calculate Precision, Recall, and F1-Score for each of the 3 classes
        const metrics = [];
        for (let c = 0; c < 3; c++) {
            let truePositives = confusionMatrix[c][c];
            let falsePositives = 0;
            let falseNegatives = 0;
            
            for (let i = 0; i < 3; i++) {
                if (i !== c) {
                    falsePositives += confusionMatrix[i][c];
                    falseNegatives += confusionMatrix[c][i];
                }
            }
            
            const precision = truePositives / (truePositives + falsePositives || 1);
            const recall = truePositives / (truePositives + falseNegatives || 1);
            const f1 = 2 * (precision * recall) / (precision + recall || 1);
            
            metrics.push({ precision, recall, f1 });
        }
        
        return {
            loss: totalLoss / samples.length,
            accuracy: correct / samples.length,
            confusionMatrix,
            metrics
        };
    }
}

/**
 * Generate synthetic training samples with realistic distributions
 * for the three classes: Alert (0), Drowsy (1), and Distracted (2).
 */
export function generateSyntheticData(sampleCountPerClass = 50) {
    const data = [];
    
    // Class 0: Alert
    // EAR high (0.26 - 0.36), MAR low (0.05 - 0.18), YawDiff low (0.0 - 0.15)
    for (let i = 0; i < sampleCountPerClass; i++) {
        const ear = 0.26 + Math.random() * 0.10;
        const mar = 0.05 + Math.random() * 0.12;
        const yawDiff = Math.random() * 0.15;
        data.push({ inputs: [ear, mar, yawDiff], target: 0 });
    }
    
    // Class 1: Drowsy
    // EAR low (0.10 - 0.19), MAR medium/high (0.08 - 0.75 yawn), YawDiff low/medium (0.0 - 0.25)
    for (let i = 0; i < sampleCountPerClass; i++) {
        const ear = 0.10 + Math.random() * 0.08;
        // half are yawning, half normal breathing
        const mar = Math.random() > 0.5 ? (0.50 + Math.random() * 0.25) : (0.08 + Math.random() * 0.15);
        const yawDiff = Math.random() * 0.25;
        data.push({ inputs: [ear, mar, yawDiff], target: 1 });
    }
    
    // Class 2: Distracted
    // EAR high/normal (0.24 - 0.36), MAR low/normal (0.05 - 0.25), YawDiff high (0.42 - 0.85)
    for (let i = 0; i < sampleCountPerClass; i++) {
        const ear = 0.24 + Math.random() * 0.12;
        const mar = 0.05 + Math.random() * 0.18;
        const yawDiff = 0.42 + Math.random() * 0.43;
        data.push({ inputs: [ear, mar, yawDiff], target: 2 });
    }
    
    // Shuffle the dataset
    return data.sort(() => Math.random() - 0.5);
}
