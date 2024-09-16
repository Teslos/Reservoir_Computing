module spikerate
using Distributions
export rate, rate_conv

function rate(
    data::AbstractArray,
    num_steps::Int64;
    gain::Float64=1.0,
    offset::Float64=0.0,
    first_spike_time::Int=0,
    time_var_input::Bool=false,
)

    """
    Spike rate encoding of input data. Convert array into Poisson spike
    trains using the features as the mean of a binomial distribution. If `num_steps`
    is specified, then the data will be repeated in the first dimension before rate encoding.

    If data is time-varying, array dimensions use time first.

    :param data: Data array for a single batch of shape [batch x input_size]
    :param num_steps: Number of time steps. Only specify if input data
        does not already have time dimension, defaults to ``false``
    :param gain: Scale input features by the gain, defaults to ``1.0``
    :param offset: Shift input features by the offset, defaults to ``0.0``
    :param first_spike_time: Time to first spike, defaults to ``0``
    :param time_var_input: Set to ``true`` if input array is time-varying.
        Otherwise, `first_spike_time!=0` will modify the wrong dimension.
        Defaults to ``false``
    :return: rate encoding spike train of input features of shape
        [num_steps x batch x input_size]
    """

    if first_spike_time < 0 || (num_steps != false && num_steps < 0)
        throw(ArgumentError("``first_spike_time`` and ``num_steps`` cannot be negative."))
    end

    if num_steps != false && first_spike_time > (num_steps - 1)
        throw(ArgumentError("``first_spike_time`` must be equal to or less than num_steps-1"))
    elseif !time_var_input && num_steps == false
        throw(ArgumentError("If the input data is time-varying, set ``time_var_input=true``.\n If the input data is not time-varying, ensure ``num_steps > 0``."))
    end

    if first_spike_time > 0 && !time_var_input && num_steps == false
        throw(ArgumentError("``num_steps`` must be specified if both the input is not time-varying and ``first_spike_time`` is greater than 0."))
    end

    if time_var_input && num_steps != false
        throw(ArgumentError("``num_steps`` should not be specified if input is time-varying, i.e., ``time_var_input=true``.\n The first dimension of the input data + ``first_spike_time`` will determine ``num_steps``."))
    end

    # intended for time-varying input data
    if time_var_input
        spike_data = rate_conv(data)

        # zeros are added directly to the start of 0th (time) dimension
        if first_spike_time > 0
            spike_data = vcat(
                zeros(append!([first_spike_time], size(spike_data)[2:end]...)),
                spike_data
            )
        end

    # intended for time-static input data
    else
        # Generate a tuple: (num_steps, 1..., 1) where the number of 1's
        # = number of dimensions in the original data.
        # Multiply by gain and add offset.
        time_data = (
            repeat(
                data,
                inner=(num_steps, fill(1, ndims(data))...)
            ) * gain .+ offset
        )
        time_data = reshape(time_data, (num_steps, size(data)...))
        spike_data = rate_conv(time_data)

        # zeros are multiplied by the start of the 0th (time) dimension
        if first_spike_time > 0
            spike_data[1:first_spike_time, :] .= 0
        end
    end

    return spike_data
end

using Flux

function rate_conv(data::AbstractArray)
    """
    Convert array into Poisson spike trains using the features as
    the mean of a binomial distribution.
    Values outside the range of [0, 1] are clipped so they can be
    treated as probabilities.

    :param data: Data array for a single batch of shape [batch x input_size]
    :return: rate encoding spike train of input features of shape
             [num_steps x batch x input_size]
    """

    # Clip all features between 0 and 1 so they can be used as probabilities.
    clipped_data = clamp.(data, 0, 1)

    # Generate spikes according to a Bernoulli distribution
    spike_data = rand.(Bernoulli.(clipped_data))

    return spike_data
end

end