class NeuralNetwork(
    private val layers: Array<Layer>
) {

    init {
        require(layers.size > 1) {
            "Requires input and exit"
        }
    }

    fun forward(input: DoubleArray): DoubleArray {

        require(input.size == layers[0].neurons.size) {
            "Input must correspond to the input layer"
        }

        var current = input

        layers
            .drop(n = 1) // drop input layer
            .forEach { layer ->
                current = layer.neurons.map {
                    it.activation(current)
                }
            }

        return current
    }

    class Layer(
        val neurons: Array<Neuron>
    )
}

private inline fun <T> Array<T>.map(transform: (T) -> Double): DoubleArray {
    return DoubleArray(this.size) { index ->
        transform(get(index))
    }
}
