INTEL_ROOT=$HOME/intel
MKL_ROOT=$INTEL_ROOT/mkl

set -x

g++ -O3 -msse3 -m64 -w -I$MKL_ROOT/include mkl_conv.cpp -Wl,--start-group "$MKL_ROOT/lib/intel64"/libmkl_intel_lp64.a "$MKL_ROOT/lib/intel64"/libmkl_intel_thread.a "$MKL_ROOT/lib/intel64"/libmkl_core.a -Wl,--end-group -L"$INTEL_ROOT/compiler/lib/intel64" -liomp5 -lpthread -ldl -lm -o mkl_conv
