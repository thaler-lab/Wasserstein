SRCS = cnpy.cpp basic_example.cpp
CXXFLAGS = -O3 -Wall -std=c++14 -g

ifeq ($(shell uname), Darwin)
	CXXFLAGS += -Xpreprocessor -fopenmp
	LIBRARIES += -lomp
endif
ifeq ($(shell uname), Linux)
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

INCLUDES  += -I../wasserstein
LIBRARIES += -lz

# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine
DEPDIR = .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(INCLUDES) -c
%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

.PHONY: all clean
all: $(SRCS:.cpp=.o) basic_example

basic_example: basic_example.o cnpy.o
	$(CXX) -o $@ $^ $(LIBRARIES) $(LDFLAGS)

clean:
	rm -fv *.o

$(DEPDIR): ; @mkdir -p $@
DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))
