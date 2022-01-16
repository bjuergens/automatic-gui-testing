def transposed_conv_output_dimension(input_size, stride, padding, dilation, kernel_size, output_padding):
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


def main():

    output_dimension = transposed_conv_output_dimension(
        input_size=224,
        stride=2,
        padding=3,
        dilation=1,
        kernel_size=7,
        output_padding=1
    )

    print(f"Output dimension: {output_dimension}")


if __name__ == "__main__":
    main()
