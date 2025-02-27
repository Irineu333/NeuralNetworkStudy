import extension.round
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuralNetworkTest {

    @Test
    fun `simulate logical port XOR`() {

        // given
        val neuralNetwork = NeuralNetwork.create(2, 4, 1)

        // training
        neuralNetwork.train(
            inputs = arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 0.0),
                doubleArrayOf(1.0, 1.0),
            ),
            expectedOutputs = arrayOf(
                doubleArrayOf(0.0),
                doubleArrayOf(1.0),
                doubleArrayOf(1.0),
                doubleArrayOf(0.0)
            ),
            epochs = 10_000,
        )

        // testing
        run {
            // 0 xor 0 = 0
            val outputs = neuralNetwork.forward(doubleArrayOf(0.0, 0.0))

            assertEquals(0.0, outputs.single().round())
        }

        run {
            // 0 xor 1 = 1
            val outputs = neuralNetwork.forward(doubleArrayOf(0.0, 1.0))

            assertEquals(1.0, outputs.single().round())
        }

        run {
            // 1 xor 0 = 1
            val outputs = neuralNetwork.forward(doubleArrayOf(1.0, 0.0))

            assertEquals(1.0, outputs.single().round())
        }

        run {
            // 1 xor 1 = 0
            val outputs = neuralNetwork.forward(doubleArrayOf(1.0, 1.0))

            assertEquals(0.0, outputs.single().round())
        }
    }
}

private fun NeuralNetwork.train(
    inputs: Array<DoubleArray>,
    expectedOutputs: Array<DoubleArray>,
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