import extension.round
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NeuronTest {

    private lateinit var neuron: Neuron

    @Test
    fun `simulate logical port AND, with STEP`() {

        // given
        neuron = Neuron(
            weights = doubleArrayOf(1.0, 1.0),
            bias = -1.5,
            activation = Neuron.Activation.STEP
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
    fun `simulate logical port OR, with STEP`() {

        // given
        neuron = Neuron(
            weights = doubleArrayOf(1.0, 1.0),
            bias = -0.5,
            activation = Neuron.Activation.STEP
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
    fun `simulate logical port AND, with SIGMOID`() {

        // given
        neuron = Neuron(
            weights = doubleArrayOf(2.0, 2.0),
            bias = -3.0,
            activation = Neuron.Activation.SIGMOID
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
    fun `simulate logical port OR, with SIGMOID`() {

        // given
        neuron = Neuron(
            weights = doubleArrayOf(2.0, 2.0),
            bias = -1.0,
            activation = Neuron.Activation.SIGMOID
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