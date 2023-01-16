interface NDArray : SizeAware, DimensionAware {
    fun at(point: Point): Int

    fun set(point: Point, value: Int)

    fun copy(): NDArray

    fun view(): NDArray

    fun add(other: NDArray)

    fun dot(other: NDArray): NDArray
}


open class DefaultNDArray private constructor(private val array: IntArray, private val shape: Shape) : NDArray {
    override val ndim: Int = shape.ndim
    override fun dim(i: Int): Int = shape.dim(i)
    override val size: Int = array.size

    companion object {
        fun ones(shape: Shape): NDArray = DefaultNDArray(IntArray(shape.size) { 1 }, shape)

        fun zeros(shape: Shape): NDArray = DefaultNDArray(IntArray(shape.size) { 0 }, shape)

        protected fun nextIndex(point: Point, shape: Shape): Point {
            var overflow = 1
            val coords = IntArray(point.ndim) { 0 }
            for (i in shape.ndim - 1 downTo 0) {
                coords[i] = (point.dim(i) + overflow) % shape.dim(i)
                overflow = (point.dim(i) + overflow) / shape.dim(i)
            }
            return DefaultPoint(*coords)
        }
    }

    private fun get1DIndex(point: Point): Int {
        if (point.ndim != ndim) {
            throw NDArrayException.IllegalPointDimensionException(point.ndim, ndim)
        }

        var index: Int = point.dim(0)
        if (index >= dim(0)) {
            throw NDArrayException.IllegalPointCoordinateException(0)
        }

        for (i in 1 until point.ndim) {
            if (point.dim(i) >= dim(i) || point.dim(i) < 0) {
                throw NDArrayException.IllegalPointCoordinateException(i)
            }
            index = dim(i) * index + point.dim(i)
        }

        return index
    }

    override fun at(point: Point): Int = array[get1DIndex(point)]

    override fun set(point: Point, value: Int) {
        array[get1DIndex(point)] = value
    }

    override fun copy(): NDArray = DefaultNDArray(array.clone(), shape)

    override fun add(other: NDArray) {
        if (other.ndim + 1 != ndim && other.ndim != ndim) {
            throw NDArrayException.NonMatchingDimensionsException(ndim, other.ndim)
        }
        for (i in 0 until other.ndim) {
            if (dim(i) != other.dim(i)) {
                throw NDArrayException.NonMatchingDimensionsException(dim(i), other.dim(i))
            }
        }

        var curIndex: Point = DefaultPoint(*IntArray(other.ndim) { 0 })
        val otherShape: Shape = DefaultShape(*(0 until other.ndim).map { other.dim(it) }.toIntArray())
        val secondDim = if (ndim != other.ndim) dim(ndim - 1) else 1
        for (i in 0 until secondDim) {
            for (j in 0 until other.size) {
                array[i + j * secondDim] += other.at(curIndex)
                curIndex = nextIndex(curIndex, otherShape)
            }
        }

    }

    override fun dot(other: NDArray): NDArray {
        if (ndim != 2 || other.ndim > 2 || other.dim(0) != dim(1)) {
            throw NDArrayException.NonMatchingDimensionsException(dim(1), other.dim(0))
        }

        val secondDim = if (other.ndim == 1) 1 else other.dim(1)
        val result = zeros(DefaultShape(dim(0), secondDim))
        var sum: Int
        for (i1 in 0 until dim(0)) {
            for (i2 in 0 until secondDim) {
                sum = 0
                for (k in 0 until dim(1)) {
                    sum += at(DefaultPoint(i1, k)) * other.at(
                        if (other.ndim == 1)
                            DefaultPoint(k) else DefaultPoint(k, i2)
                    )
                }
                result.set(DefaultPoint(i1, i2), sum)
            }
        }
        return result
    }

    override fun view(): NDArray {
        return ModifiedNDArray()
    }

    internal inner class ModifiedNDArray: NDArray by this@DefaultNDArray
}

sealed class NDArrayException(val mes: String) : Exception(mes) {
    class IllegalPointCoordinateException(val pos: Int) :
        NDArrayException("Index out of range at position $pos")

    class IllegalPointDimensionException(val d1: Int, val d2: Int) :
        NDArrayException("Index does not match Array dimension: $d1 vs. $d2")

    class NonMatchingDimensionsException(val d1: Int, val d2: Int) :
        NDArrayException("Arrays do not have matching dimensions: $d1 vs. $d2")
}