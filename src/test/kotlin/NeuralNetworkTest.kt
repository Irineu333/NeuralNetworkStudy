import extension.round
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuralNetworkTest {

    @Test
    fun `simulate logical port XOR`() {

        // given

        val inputLayer = NeuralNetwork.Layer(neurons = Array(2) { Neuron() })

        val hiddenLayer = NeuralNetwork.Layer(
            neurons = arrayOf(
                Neuron(
                    bias = 1.3851242885537485,
                    weights = doubleArrayOf(
                        -0.34281371276528644,
                        -0.5962433104446242
                    )
                ),
                Neuron(
                    bias = 2.8474964898137056,
                    weights = doubleArrayOf(
                        5.5597694337117805,
                        -5.497819094902079
                    )
                ),
                Neuron(
                    bias = -1.3181452901046906,
                    weights = doubleArrayOf(
                        1.7858333842169294,
                        -1.5281806427286417
                    )
                ),
                Neuron(
                    bias = -2.5632916201118467,
                    weights = doubleArrayOf(
                        4.5726559437031185,
                        -4.694734199962801
                    )
                )
            )
        )

        val outputLayer = NeuralNetwork.Layer(
            neurons = Array(1) {
                Neuron(
                    bias = 2.4757123239361394,
                    weights = doubleArrayOf(
                        1.196265565534793,
                        -7.837060755046451,
                        2.367324599340193,
                        6.9315813360259115
                    )
                )
            }
        )

        val neuralNetwork = NeuralNetwork(
            layers = arrayOf(inputLayer, hiddenLayer, outputLayer),
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
