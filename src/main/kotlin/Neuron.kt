import kotlin.math.exp

/**
 * An artificial neuron of the perceptron type.
 */
class Neuron(
    private val weights: DoubleArray = DoubleArray(size = 0),
    private val bias: Double = 0.0,
    private val activation: Activation = Activation.SIGMOID
) {
    fun activation(
        inputs: DoubleArray
    ): Double {

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
        RELU,
        STEP // Boolean
    }

    private fun relu(x: Double): Double = if (x < 0.0) 0.0 else x

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

    private fun step(x: Double) = if (x >= 0) 1.0 else 0.0
}
