#BUILD AND RUN
as ./Quplexity/ARM/gates.s -o ./Quplexity/ARM/gates.o
gcc -shared -o ./Quplexity/ARM/gates.so ./Quplexity/ARM/gates.o 
#gcc -shared -o ./Quplexity/bus.so -fPIC ./Quplexity/bus.c
python3 main.py