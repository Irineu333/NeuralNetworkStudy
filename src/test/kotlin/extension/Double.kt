package extension

import java.math.BigDecimal
import java.math.RoundingMode

fun Double.round(
    scale: Int = 0,
    mode: RoundingMode = RoundingMode.HALF_UP
) = BigDecimal(toString())
    .setScale(scale, mode)
    .toDouble()
