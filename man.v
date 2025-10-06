module float_add(
    input  wire [7:0] aIn,
    input  wire [7:0] bIn,
    output reg  [7:0] result
);

    // --- Intermediate Signals ---
    
    // Wires for the ordered inputs from big_number_first
    wire [7:0] big_num;
    wire [7:0] small_num;
    
    // Wires for the components of the ordered numbers
    wire [2:0] exp_big;
    wire [4:0] mant_big;
    wire [2:0] exp_small;
    wire [4:0] mant_small;
    
    // Wire for the difference between exponents (the shift distance)
    wire [2:0] exp_diff;
    
    // Wire for the smaller mantissa after being shifted for alignment
    wire [4:0] mant_small_shifted;
    
    // Wires for the result of the mantissa addition
    wire [4:0] sum_mant;
    wire       cout_mant;

    
    // --- Submodule Instantiations ---

    // 1. Order the numbers to ensure 'big_num' has the larger or equal exponent
    big_number_first order_inst (
        .aIn(aIn),
        .bIn(bIn),
        .aOut(big_num),
        .bOut(small_num)
    );

    // 2. Deconstruct the ordered numbers into their exponent and mantissa parts
    assign exp_big   = big_num[7:5];
    assign mant_big  = big_num[4:0];
    assign exp_small = small_num[7:5];
    assign mant_small = small_num[4:0];

    // 3. Calculate the difference in exponents for the shift
    assign exp_diff = exp_big - exp_small;

    // 4. Align the smaller mantissa by shifting it right
    shifter shift_inst (
        .in(mant_small),
        .distance(exp_diff),
        .direction(1'b1), // 1 = right shift
        .out(mant_small_shifted)
    );

    // 5. Add the aligned mantissas
    adder adder_inst (
        .a(mant_big),
        .b(mant_small_shifted),
        .sum(sum_mant),
        .cout(cout_mant)
    );

    
    // --- Final Result Logic ---

    // 6. Normalize and saturate the final result based on the adder's output
    always @(*) begin
        if (cout_mant == 1'b1) begin
            // Mantissa addition overflowed
            if (exp_big == 3'b111) begin
                // Exponent is already maxed out, so saturate the output
                [cite_start]result = 8'b111_11111; [cite: 123, 124]
            end else begin
                // Increment the exponent and take the top 5 bits of the 6-bit sum
                [cite_start]result = {exp_big + 1, {cout_mant, sum_mant[4:1]}}; [cite: 120, 121]
            end
        end else begin
            // No overflow, the result is already normalized
            result = {exp_big, sum_mant};
        end
    end

endmodule
