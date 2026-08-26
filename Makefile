NVCC = nvcc
CXX_STD = -std=c++17
NVCC_FLAGS = $(CXX_STD) -O2
DEBUG_FLAGS = $(CXX_STD) -g -G -O0

SRCS_CU = backward.cu forward.cu
SRCS_CPP = main.cpp network.cpp tensor.cpp

OBJS = $(SRCS_CU:.cu=.o) $(SRCS_CPP:.cpp=.o)
DEBUG_OBJS = $(SRCS_CU:.cu=.debug.o) $(SRCS_CPP:.cpp=.debug.o)

TARGET = cudalib
DEBUG_TARGET = cudalib_debug

all: $(TARGET)

debug: $(DEBUG_TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(NVCC) $(DEBUG_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.debug.o: %.cu
	$(NVCC) $(DEBUG_FLAGS) -c $< -o $@

%.debug.o: %.cpp
	$(NVCC) $(DEBUG_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(DEBUG_OBJS) $(TARGET) $(DEBUG_TARGET)

.PHONY: all debug clean