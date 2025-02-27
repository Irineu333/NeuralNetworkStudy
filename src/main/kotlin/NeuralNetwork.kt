class NeuralNetwork(
    private val layers: Array<Layer>
) {

    init {
        require(layers.size > 1) {
            "Requires input and exit"
        }
    }

    fun forward(inputs: DoubleArray): DoubleArray {

        require(inputs.size == layers[0].neurons.size) {
            "Input must correspond to the input layer"
        }

        var current = inputs

        layers
            .drop(n = 1) // drop input layer
            .forEach { layer ->
                current = layer.neurons.map {
                    it.activation(current)
                }
            }

        return current
    }

    fun train(
        inputs: DoubleArray,
        expectedOutput: DoubleArray,
        learningRate: Double = 0.1,
    ) {
        val layers = layers.drop(n = 1)

        // Forward
        val resultsByLayer = mutableListOf(inputs)

        var current = inputs

        layers.forEach { layer ->
            current = layer.neurons.map {
                it.activation(current)
            }

            resultsByLayer.add(current)
        }

        // Calculate output layer errors
        val outputLayerErrors = resultsByLayer.last().let {
            DoubleArray(it.size).apply {
                it.forEachIndexed { index, result ->
                    this[index] = expectedOutput[index] - result
                }
            }
        }

        // Backward
        val errorsByLayer = mutableListOf(outputLayerErrors)

        for ((layer, nextLayer) in layers.zipWithNext().reversed()) {
            val errorsByNeuron = errorsByLayer.first()
            val errors = DoubleArray(layer.neurons.size)

            for (neuronIndex in layer.neurons.indices) {
                for (nextNeuronIndex in nextLayer.neurons.indices) {
                    val nextLayerNeuron = nextLayer.neurons[nextNeuronIndex]
                    val nextLayerWeight = nextLayerNeuron.weights[neuronIndex]

                    errors[neuronIndex] += errorsByNeuron[nextNeuronIndex] * nextLayerWeight
                }
            }

            errorsByLayer.add(index = 0, errors)
        }

        // Update weights and biases
        for ((layerIndex, layer) in layers.withIndex()) {

            val errorsByNeuron = errorsByLayer[layerIndex]

            for (neuronIndex in layer.neurons.indices) {

                val neuron = layer.neurons[neuronIndex]
                val output = resultsByLayer[layerIndex + 1][neuronIndex]

                // Calculate derivative of activation function
                val derivative = when (neuron.activation) {
                    Neuron.Activation.SIGMOID -> output * (1 - output)
                    Neuron.Activation.RELU -> if (output > 0) 1.0 else 0.0
                    Neuron.Activation.STEP -> 1.0 // Approximation
                }

                val delta = errorsByNeuron[neuronIndex] * derivative

                val resultsByNeuron = resultsByLayer[layerIndex]

                // Update weights
                resultsByNeuron.withIndex().forEach { (index, result) ->
                    neuron.weights[index] += learningRate * delta * result
                }

                // Update bias
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
