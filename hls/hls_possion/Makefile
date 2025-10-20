# Simple C Makefile

CC = g++
CFLAGS = -Wall -g
TARGET = simple_test
SOURCES = simple_test.cpp simple_test_tb.cpp

# Build executable
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

# Run program
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET) $(TARGET).exe *.log
.PHONY: run clean