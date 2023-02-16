
# compile but do not link
g++ -lgecodedriver -lgecodesupport -lgecodekernel  -c money.cpp

# finish compilation by linking
g++ -o money money.o -lgecodesearch -lgecodeint -lgecodekernel -lgecodesupport -lgecodegist -lgecodeminimodel -lgecodeflatzinc -lgecodedriver

# compile in 1 command:
g++ money.cpp -lgecodedriver -lgecodesupport -lgecodekernel -lgecodesearch -lgecodeint  -lgecodegist -lgecodeminimodel -lgecodeflatzinc -o money

# run
./money




SEND+MORE=MONEY
        {9, 5, 6, 7, 1, 0, 8, 2}

Initial
        propagators: 2
        branchers:   1

Summary
        runtime:      0.014 (14.522 ms)
        solutions:    1
        propagations: 14
        nodes:        7
        failures:     3
        restarts:     0
        no-goods:     0
        peak depth:   1



g++ tsp.cpp -lgecodedriver -lgecodesupport -lgecodekernel -lgecodesearch -lgecodeint  -lgecodegist -lgecodeminimodel -lgecodeflatzinc -o tsp
