import kotlin.math.exp

/**
 * An artificial neuron of the perceptron type.
 */
class Neuron(
    val weights: DoubleArray = DoubleArray(size = 0),
    val activation: Activation = Activation.SIGMOID,
    var bias: Double = 0.0
) {
    fun activation(
        inputs: DoubleArray
    ): Double {

        require(inputs.size == weights.size) {
            "Input size must match weights size"
        }

        val weightedSum = inputs
            .zip(weights)
            .sumOf { (input, weight) -> input * weight } + bias

        return when (activation) {
            Activation.SIGMOID -> sigmoid(weightedSum)
            Activation.RELU -> relu(weightedSum)
            Activation.STEP -> step(weightedSum)
        }
    }

    enum class Activation {
        SIGMOID, // Probability
        RELU, // Deep learning
        STEP // Boolean
    }

    private fun relu(x: Double): Double = if (x < 0.0) 0.0 else x

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

    private fun step(x: Double) = if (x >= 0) 1.0 else 0.0

    companion object {
        fun random(
            weights: Int,
            activation: Activation = Activation.SIGMOID
        ) = Neuron(
            weights = DoubleArray(weights) { Math.random() },
            bias = Math.random(),
            activation = activation
        )
    }
}
