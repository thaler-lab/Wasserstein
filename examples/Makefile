CXX = g++
SRCS = $(shell ls *.cpp) cnpy.cpp
CXXFLAGS = -O3 -Wall -std=c++14 -g -ffast-math

ifeq ($(shell uname), Darwin)
	CXXFLAGS += -Xpreprocessor -fopenmp
	LIBRARIES += -lomp
endif
ifeq ($(shell uname), Linux)
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

INCLUDES  += -I../wasserstein -Iinclude
LIBRARIES += -lz

# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine
DEPDIR = .deps
COMPILE.cpp = $(CXX) $(CXXFLAGS) $(INCLUDES) -c $(<) -o $(@) -MT $(@) -MMD -MP -MF $(DEPDIR)/$*.d
%.o : %.cpp
src/%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp)
src/%.o : src/%.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp)

.PHONY: all clean
all: emd_example pairwise_emds_example theory_space_example

emd_example: src/emd_example.o src/cnpy.o
	$(CXX) -o $@ $^ $(LIBRARIES) $(LDFLAGS)

pairwise_emds_example: src/pairwise_emds_example.o src/cnpy.o
	$(CXX) -o $@ $^ $(LIBRARIES) $(LDFLAGS)

serialization_example: src/serialization_example.o src/cnpy.o
	$(CXX) -o $@ $^ $(LIBRARIES) $(LDFLAGS) -lboost_serialization

theory_space_example: src/theory_space_example.o src/cnpy.o
	$(CXX) -o $@ $^ $(LIBRARIES) $(LDFLAGS)

clean:
	rm -rfv *.o *_example src/*.o $(DEPDIR)

$(DEPDIR): ; @mkdir -p $@
DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))
