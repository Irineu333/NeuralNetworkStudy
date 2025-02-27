class NeuralNetwork(
    private val layers: Array<Layer>
) {

    private val inputLayers get() = layers.first()
    private val hiddenLayers get() = layers.drop(n = 1).dropLast(n = 1)
    private val outputLayers get() = layers.last()
    private val actionableLayers get() = layers.drop(n = 1)

    init {
        require(layers.size > 1) {
            "Requires input and output layers"
        }
    }

    fun forward(inputs: DoubleArray): DoubleArray {

        require(inputs.size == layers[0].neurons.size) {
            "Input must correspond to the input layer"
        }

        return actionableLayers.fold(inputs) { input, layer ->
            layer.neurons.map {
                it.activation(input)
            }
        }
    }

    fun train(
        inputs: DoubleArray,
        expectedOutput: DoubleArray,
        learningRate: Double = 0.1,
    ) {
        // Forward
        val resultsByLayer = mutableListOf(inputs)

        actionableLayers.fold(inputs) { currentInput, layer ->
            layer.neurons.map {
                it.activation(currentInput)
            }.also {
                resultsByLayer.add(it)
            }
        }

        // Calculate output layer errors
        val outputLayerErrors = DoubleArray(outputLayers.neurons.size) { index ->
            expectedOutput[index] - resultsByLayer.last()[index]
        }

        // Backpropagation
        val errorsByLayer = mutableListOf(outputLayerErrors)

        actionableLayers.zipWithNext().reversed().forEach { (layer, nextLayer) ->
            val currentErrors = errorsByLayer.first()

            val layerErrors = DoubleArray(layer.neurons.size) { neuronIndex ->
                nextLayer.neurons.indices.sumOf { nextNeuronIndex ->
                    currentErrors[nextNeuronIndex] * nextLayer.neurons[nextNeuronIndex].weights[neuronIndex]
                }
            }

            errorsByLayer.add(index = 0, layerErrors)
        }

        // Update weights and biases
        actionableLayers.forEachIndexed { layerIndex, layer ->
            val errors = errorsByLayer[layerIndex]
            val outputs = resultsByLayer[layerIndex + 1]

            layer.neurons.forEachIndexed { neuronIndex, neuron ->
                val error = errors[neuronIndex]
                val output = outputs[neuronIndex]

                val derivative = when (neuron.activation) {
                    Neuron.Activation.SIGMOID -> output * (1 - output)
                    Neuron.Activation.RELU -> if (output > 0) 1.0 else 0.01 // Leaky ReLU
                    Neuron.Activation.STEP -> error("Does not support backpropagation")
                }

                val delta = error * derivative

                resultsByLayer[layerIndex].forEachIndexed { inputIndex, input ->
                    neuron.weights[inputIndex] += learningRate * delta * input
                }

                neuron.bias += learningRate * delta
            }
        }
    }

    class Layer(
        val neurons: Array<Neuron>
    )

    companion object {
        fun create(
            vararg layers: Int,
            activation: Neuron.Activation = Neuron.Activation.SIGMOID,
        ): NeuralNetwork {
            return NeuralNetwork(
                layers = Array(layers.size) { layer ->
                    Layer(
                        neurons = Array(layers[layer]) {
                            if (layer == 0) {
                                Neuron() // input node
                            } else {
                                Neuron.random(
                                    weights = layers[layer - 1],
                                    activation = activation
                                )
                            }
                        }
                    )
                }
            )
        }
    }
}

private inline fun <T> Array<T>.map(
    transform: (T) -> Double
): DoubleArray {
    return DoubleArray(size) { index ->
        transform(get(index))
    }
}
