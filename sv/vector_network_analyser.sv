module vector_network_analyser( input reset,
                 input A, 
                 output Z );
//reset does nothing
assign Z= !A;

endmodule
