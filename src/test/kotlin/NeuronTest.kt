import extension.round
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuronTest {

    private lateinit var neuron: Neuron

    @Test
    fun `train to simulate a logical port and AND, with STEP`() {

        // given
        neuron = Neuron.random(
            weights = 2,
            activation = Neuron.Activation.STEP
        )

        // training
        neuron.train(
            inputs = arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(1.0, 1.0),
            ),
            expectedOutputs = arrayOf(0.0, 0.0, 0.0, 1.0),
            epochs = 10,
        )

        // testing
        run {
            // 0 and 0 = 0
            val output = neuron.activation(doubleArrayOf(0.0, 0.0))

            assertEquals(0.0, output)
        }

        run {
            // 0 and 1 = 0
            val outputs = neuron.activation(doubleArrayOf(0.0, 1.0))

            assertEquals(0.0, outputs)
        }

        run {
            // 1 and 0 = 0
            val outputs = neuron.activation(doubleArrayOf(1.0, 0.0))

            assertEquals(0.0, outputs)
        }

        run {
            // 1 and 1 = 1
            val outputs = neuron.activation(doubleArrayOf(1.0, 1.0))

            assertEquals(1.0, outputs)
        }
    }

    @Test
    fun `train to simulate a logical port OR, with STEP`() {

        // given
        neuron = Neuron.random(
            weights = 2,
            activation = Neuron.Activation.STEP
        )

        // training
        neuron.train(
            inputs = arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(1.0, 1.0),
            ),
            expectedOutputs = arrayOf(0.0, 1.0, 1.0, 1.0),
            epochs = 10,
        )

        // testing
        run {
            // 0 or 0 = 0
            val outputs = neuron.activation(doubleArrayOf(0.0, 0.0))

            assertEquals(0.0, outputs)
        }

        run {
            // 0 or 1 = 1
            val outputs = neuron.activation(doubleArrayOf(0.0, 1.0))

            assertEquals(1.0, outputs)
        }

        run {
            // 1 or 0 = 1
            val outputs = neuron.activation(doubleArrayOf(1.0, 0.0))

            assertEquals(1.0, outputs)
        }

        run {
            // 1 or 1 = 1
            val outputs = neuron.activation(doubleArrayOf(1.0, 1.0))

            assertEquals(1.0, outputs)
        }
    }

    @Test
    fun `train to simulate a logical port AND, with SIGMOID`() {

        // given
        neuron = Neuron.random(
            weights = 2,
            activation = Neuron.Activation.SIGMOID
        )

        // training
        neuron.train(
            inputs = arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(1.0, 1.0),
            ),
            expectedOutputs = arrayOf(0.0, 0.0, 0.0, 1.0),
            epochs = 100,
        )

        // testing
        run {
            // 0 and 0 = 0
            val output = neuron.activation(doubleArrayOf(0.0, 0.0))

            assertEquals(0.0, output.round())
        }

        run {
            // 0 and 1 = 0
            val outputs = neuron.activation(doubleArrayOf(0.0, 1.0))

            assertEquals(0.0, outputs.round())
        }

        run {
            // 1 and 0 = 0
            val output = neuron.activation(doubleArrayOf(1.0, 0.0))

            assertEquals(0.0, output.round())
        }

        run {
            // 1 and 1 = 1
            val outputs = neuron.activation(doubleArrayOf(1.0, 1.0))

            assertEquals(1.0, outputs.round())
        }
    }

    @Test
    fun `train to simulate a logical port OR, with SIGMOID`() {

        // given
        neuron = Neuron.random(
            weights = 2,
            activation = Neuron.Activation.SIGMOID
        )

        // training
        neuron.train(
            inputs = arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(1.0, 1.0),
            ),
            expectedOutputs = arrayOf(0.0, 1.0, 1.0, 1.0),
            epochs = 100,
        )

        // testing
        run {
            // 0 or 0 = 0
            val outputs = neuron.activation(doubleArrayOf(0.0, 0.0))

            assertEquals(0.0, outputs.round())
        }

        run {
            // 0 or 1 = 1
            val outputs = neuron.activation(doubleArrayOf(0.0, 1.0))

            assertEquals(1.0, outputs.round())
        }

        run {
            // 1 or 0 = 1
            val output = neuron.activation(doubleArrayOf(1.0, 0.0))

            assertEquals(1.0, output.round())
        }

        run {
            // 1 or 1 = 1
            val outputs = neuron.activation(doubleArrayOf(1.0, 1.0))

            assertEquals(1.0, outputs.round())
        }
    }

    @Test
    fun `train to simulate a logical port AND, with RELU`() {

        // given
        neuron = Neuron.random(
            weights = 2,
            activation = Neuron.Activation.RELU
        )

        // training
        neuron.train(
            inputs = arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(1.0, 1.0),
            ),
            expectedOutputs = arrayOf(0.0, 0.0, 0.0, 1.0),
            epochs = 10,
        )

        // testing
        run {
            // 0 and 0 = 0
            val output = neuron.activation(doubleArrayOf(0.0, 0.0))

            assertEquals(0.0, output.round())
        }

        run {
            // 0 and 1 = 0
            val outputs = neuron.activation(doubleArrayOf(0.0, 1.0))

            assertEquals(0.0, outputs.round())
        }

        run {
            // 1 and 0 = 0
            val output = neuron.activation(doubleArrayOf(1.0, 0.0))

            assertEquals(0.0, output.round())
        }

        run {
            // 1 and 1 = 1
            val outputs = neuron.activation(doubleArrayOf(1.0, 1.0))

            assertEquals(1.0, outputs.round())
        }
    }

    @Test
    fun `train to simulate a logical port OR, with RELU`() {

        // given
        neuron = Neuron.random(
            weights = 2,
            activation = Neuron.Activation.RELU
        )

        // training
        neuron.train(
            inputs = arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(1.0, 1.0),
            ),
            expectedOutputs = arrayOf(0.0, 1.0, 1.0, 1.0),
            epochs = 10,
        )

        // testing
        run {
            // 0 or 0 = 0
            val outputs = neuron.activation(doubleArrayOf(0.0, 0.0))

            assertEquals(0.0, outputs.round())
        }

        run {
            // 0 or 1 = 1
            val outputs = neuron.activation(doubleArrayOf(0.0, 1.0))

            assertEquals(1.0, outputs.round())
        }

        run {
            // 1 or 0 = 1
            val output = neuron.activation(doubleArrayOf(1.0, 0.0))

            assertEquals(1.0, output.round())
        }

        run {
            // 1 or 1 = 1
            val outputs = neuron.activation(doubleArrayOf(1.0, 1.0))

            assertEquals(1.0, outputs.round())
        }
    }
}

private fun Neuron.train(
    inputs: Array<DoubleArray>,
    expectedOutputs: Array<Double>,
    epochs: Int,
    learningRate: Double = 0.1,
) {
    repeat(epochs) {
        inputs
            .zip(expectedOutputs)
            .forEach { (input, expectedOutput) ->
                train(
                    inputs = input,
                    expectedOutput = expectedOutput,
                    learningRate = learningRate
                )
            }
    }
}

private fun Neuron.train(
    inputs: DoubleArray,
    expectedOutput: Double,
    learningRate: Double
) {
    // Forward
    val calculatedOutput = activation(inputs)

    // Calculate the error
    val error = expectedOutput - calculatedOutput

    // Update weights
    inputs.forEachIndexed { index, input ->
        weights[index] += learningRate * error * input
    }

    // Update bias
    bias += learningRate * error
}
