import kotlin.math.exp

/**
 * An artificial neuron of the perceptron type.
 */
class Neuron(
    val weights: DoubleArray = DoubleArray(size = 0),
    val activation: Activation = Activation.LINEAR,
    var bias: Double = 0.0
) {

    private val DoubleArray.value
        get() = zip(weights).sumOf { (input, weight) -> input * weight } + bias

    fun activation(
        inputs: DoubleArray
    ): Double {

        require(inputs.size == weights.size) {
            "Input size must match weights size"
        }

        return when (activation) {
            Activation.SIGMOID -> sigmoid(inputs.value)
            Activation.RELU -> relu(inputs.value)
            Activation.STEP -> step(inputs.value)
            Activation.LINEAR -> inputs.single()
        }
    }

    enum class Activation {
        SIGMOID, // Probability
        RELU, // Deep learning
        STEP, // Boolean
        LINEAR
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
