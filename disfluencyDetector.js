// disfluencyDetector.js
class DisfluencyDetector {
    constructor() {
        this.model = null;
        this.isLoaded = false;
    }

    async loadModel(modelUrl) {
        try {
            // Load TFLITE model using TF.js TFLITE task API
            this.model = await tflite.loadTFLiteModel(modelUrl);
            this.isLoaded = true;
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    async processSentences(sentences) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }

        try {
            const results = await Promise.all(sentences.map(async (sentence) => {
                const tokens = sentence.split(' ');
                const inputTensor = this.preprocessSentence(tokens);
                
                // Run inference with TFLITE model
                const outputTensor = await this.model.predict(inputTensor);
                const results = await this.postprocessPrediction(outputTensor, tokens);
                
                return {
                    original: sentence,
                    tokens: tokens,
                    disfluencies: results
                };
            }));

            return results;
        } catch (error) {
            console.error('Error processing sentences:', error);
            throw error;
        }
    }

    preprocessSentence(tokens) {
        const maxLength = 512;
        const padded = tokens.slice(0, maxLength).concat(
            Array(Math.max(0, maxLength - tokens.length)).fill('[PAD]')
        );
        
        // Convert to Int32 tensor for TFLITE
        return tf.tensor2d([padded], [1, maxLength], 'int32');
    }

    async postprocessPrediction(prediction, tokens) {
        const predictionArray = await prediction.data();
        
        return tokens.map((token, idx) => ({
            token: token,
            isDisfluency: predictionArray[idx] > 0.5
        }));
    }
}