#include <cassert>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <sys/time.h>

#include "mkl.h"

using namespace std;

#define LOW 16ul
#define HIGH 1024ul
  
const size_t T = 15;
const size_t L = 5;

const size_t dimension = 4;
const size_t iw = 10;
const size_t ih = 10;
const size_t ow = 10;
const size_t oh = 10;
const size_t kw = 3;
const size_t kh = 3;

double TestConv(size_t batch_size, size_t in_channel, size_t out_channel) {

    size_t n = batch_size;
    dnnPrimitive_t convolutionFwd = static_cast<dnnPrimitive_t>(NULL);;
    size_t ic = in_channel;
    size_t oc = out_channel;
    size_t bdata_sizes[4] = { iw, ih, ic, n };
    size_t tdata_sizes[4] = { ow, oh, oc, n };
    size_t fdata_sizes[5] = { kw, kh, ic, oc, 1 };
    size_t convolutionStrides[2] = { 1, 1 };
    int inputOffset[2] = { -1, -1 };

    size_t bdata_strides[4] = { 1, iw, iw * ih, iw * ih * ic };
    size_t tdata_strides[4] = { 1, ow, ow * oh, ow * oh * oc };
    size_t fdata_strides[5] = { 1, kw, kw * kh, kw * kh * ic, kw * kh * ic / oc };

    dnnConvolutionCreateForward_F64(&convolutionFwd,
                                    NULL,
                                    dnnAlgorithmConvolutionDirect,
                                    dimension,
                                    bdata_sizes,
                                    tdata_sizes,
                                    fdata_sizes,
                                    convolutionStrides,
                                    inputOffset,
                                    dnnBorderZeros);
    
    dnnLayout_t sLayout;
    dnnLayout_t fLayout;
    dnnLayout_t dLayout;
    dnnLayoutCreate_F64(&sLayout, dimension, bdata_sizes, bdata_strides);
    dnnLayoutCreate_F64(&fLayout, dimension, fdata_sizes, fdata_strides);
    dnnLayoutCreate_F64(&dLayout, dimension, tdata_sizes, tdata_strides);
    
    void *src, *filter, *dst;
    dnnAllocateBuffer_F64(&src, sLayout);
    dnnAllocateBuffer_F64(&filter, fLayout);
    dnnAllocateBuffer_F64(&dst, dLayout);
    
    void *res_convolutionFwd[dnnResourceNumber];
    res_convolutionFwd[dnnResourceSrc] = src;
    res_convolutionFwd[dnnResourceFilter] = filter;
    res_convolutionFwd[dnnResourceDst] = dst;
    
    timeval start, end;
    for (size_t t = 0; t < T + L; ++t) {
        if (t == L) {
            gettimeofday(&start, 0);
        }
        dnnExecute_F64(convolutionFwd, res_convolutionFwd);   
    }
    gettimeofday(&end, 0);
    double milli = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * .001;
    
    dnnReleaseBuffer_F64(src);
    dnnReleaseBuffer_F64(filter);
    dnnReleaseBuffer_F64(dst);

    dnnLayoutDelete_F64(sLayout);
    dnnLayoutDelete_F64(fLayout);
    dnnLayoutDelete_F64(dLayout);

    dnnDelete_F64(convolutionFwd);

    return milli / T;
}

int main() {
    ofstream fout("rst.tsv");

    for (size_t flops = 32ul * 32 * 32; flops <= 1024ul* 1024 * 1024; flops *= 2) {
        fout << "FLOPS:\t" << flops << endl;
        fout << "b \\ ci\t";
        for (size_t n = LOW; n <= HIGH; n *= 2) {
            fout << n << "\t";
        }
        fout << endl;
        for (size_t b = LOW; b <= HIGH; b *= 2) {
            fout << b << "\t";
            for (size_t ci = LOW; ci <= HIGH; ci *= 2) {
                size_t co = flops / b / ci;
                if (co == 0 || b * co > HIGH * HIGH || ci * co > HIGH * HIGH) {
                    fout << -1 << "\t";
                    continue;
                }
                cout << "Test flops=" << flops << " b=" << b << " ci=" << ci << " co=" << co << endl;
                double time = TestConv(b, ci, co);
                fout << time << "\t";
            }
            fout << endl;
        }
        fout << endl;
    }

    return 0;
}
