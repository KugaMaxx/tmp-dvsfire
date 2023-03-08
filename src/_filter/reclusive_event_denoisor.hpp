#include "kore.hpp"
#include "denoisor.hpp"

#include <math.h>
#include <time.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <immintrin.h>

#include <cblas.h>
#include <dv-sdk/module.hpp>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

namespace edn {

class ReclusiveEventDenoisor : public EventDenoisor {
public:
    float_t sigmaT    = 1.2;
    float_t sigmaS    = 1.0;
    float_t threshold = 0.7;

    static const size_t _POLES_  = 4;
    static const size_t _THREAD_ = 8;

    float_t *Xt;
    float_t *Yt;
    float_t *Ut;

    int64_t lastTimestamp = INT64_MAX;
    float_t samplarT = 1000.;
    float_t A[_POLES_ * _POLES_] = {
        -0.348174256797851, -0.101759749190219, -0.0132846052724056, -0.000841996822694814, 
        1.               ,  0.               ,  0.                ,  0.                  ,
        0.               ,  1.               ,  0.                ,  0.                  ,
        0.               ,  0.               ,  1.                ,  0.                  ,
    };
    float_t B[_POLES_ * 1      ] = {
        1.               ,  0.               ,  0.                ,  0.                  ,
    };
    float_t C[1       * _POLES_] = {
        -0.034780277057838,  0.036136895953355, -0.0166748570101696,  0.005317422164085240,
    };
    float_t expmA[_POLES_ * _POLES_] = {
        0.663969918056621, -0.090449028116203, -0.0113960589465566, -0.000698661633195582,
        0.829767541116738,  0.952873614999921, -0.0060120912459797, -0.000372924694966144,
        0.442905109514069,  0.983975698453770,  0.9979435278591400, -0.000128271692953740,
        0.152342252959110,  0.495946760217017,  0.9994780079059620,  0.999967334556010000,
    };
    float_t expmAB[1      * _POLES_];

    float_t a0 = 1.6800, a1 = -0.6803;
    float_t b0 = 3.7350, b1 = -0.2598;
    float_t w0 = 0.6319, w1 =  1.9970;
    float_t k0 = -1.783, k1 = -1.7230;

    float_t n0, n1, n2, n3;
    float_t d1, d2, d3, d4;
    float_t m1, m2, m3, m4;

    void setCoefficient() {
        /* initialize state space filter parameters */
        samplarT = samplarT * (sigmaT / 1.);
        for (size_t i = 0; i < _POLES_; i++) {
            expmAB[i] = *(expmA + i * _POLES_);
        }

        /* initialize deriche blur filter parameters */
        float_t scale = 1.0000 / (sqrt(2 * M_PI) * sigmaS);
        a0 *= scale, a1 *= scale, b0 *= scale, b1 *= scale;

        n0 = a1 + a0;
        n1 = exp(k1/sigmaS)          * (b1*sin(w1/sigmaS)-(a1+2*a0)*cos(w1/sigmaS)) + exp(k0/sigmaS) * (b0*sin(w0/sigmaS) - (a0+2*a1) * cos(w0/sigmaS));
        n2 = 2 * exp((k0+k1)/sigmaS) * ((a0+a1)*cos(w1/sigmaS)*cos(w0/sigmaS) - b0*cos(w1/sigmaS)*sin(w0/sigmaS) - b1*cos(w0/sigmaS)*sin(w1/sigmaS)) + a1 * exp(2*k0/sigmaS) + a0 * exp(2*k1/sigmaS);
        n3 = exp((k1+2*k0)/sigmaS)   * (b1*sin(w1/sigmaS)-a1*cos(w1/sigmaS)) + exp((k0+2*k1)/sigmaS) * (b0*sin(w0/sigmaS) - a0*cos(w0/sigmaS));

        d1 = -2 * exp(k1/sigmaS)        * cos(w1/sigmaS)                    - 2 * exp(k0/sigmaS) * cos(w0/sigmaS);
        d2 =  4 * exp((k0+k1)/sigmaS)   * cos(w1/sigmaS) * cos(w0/sigmaS)   + exp(2*k1/sigmaS) + exp(2*k0/sigmaS);
        d3 = -2 * exp((k0+2*k1)/sigmaS) * cos(w1/sigmaS)                    - 2 * exp((k1+2*k0)/sigmaS) * cos(w1/sigmaS);
        d4 =  1 * exp(2*(k0+k1)/sigmaS);

        m1 = n1 - d1 * n0;
        m2 = n2 - d2 * n0;
        m3 = n3 - d3 * n0;
        m4 = -d4 * n0;
    };
    
    void updateStateSpace(float *Xt, float *Yt, float *Ut) {
        float *expmABU = (float *) calloc(_POLES_ * _LENGTH_, sizeof(float));
        
        // Yt = C * Xt
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, _LENGTH_, _POLES_, 1., C, _POLES_, Xt, _LENGTH_, 0., Yt, _LENGTH_);
        
        // expmABU = expm(A) * B * Ut
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    _POLES_, _LENGTH_, 1, 1., expmAB, 1, Ut, _LENGTH_, 0., expmABU, _LENGTH_);

        // expmABU = expm(A) * Xt + expmABU
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    _POLES_, _LENGTH_, _POLES_, 1., expmA, _POLES_, Xt, _LENGTH_, 1., expmABU, _LENGTH_);
        
        // after-processing
        cblas_scopy(_POLES_ * _LENGTH_, expmABU, 1, Xt, 1);
        memset(Ut, 0, _LENGTH_ * sizeof(*Ut));
        free(expmABU);

        return;
    };
    
    void fastDericheBlur(float *Yt) {
        int32_t *index = (int32_t *) calloc(_THREAD_, sizeof(int32_t));
        float_t *tmpRe = (float_t *) calloc(_THREAD_, sizeof(float_t));
        float_t *tmpYt = (float_t *) calloc(_LENGTH_, sizeof(float_t));

        __m256i mIndex;
        __m256 mPrevIn1, mPrevIn2, mPrevIn3, mPrevIn4;
        __m256 mPrevOut1, mPrevOut2, mPrevOut3, mPrevOut4;

        __m256 mCurIn, mSumIn, mCurOut, mSumOut;
        __m256 mSumN0, mSumN1, mSumN2, mSumN3;
        __m256 mSumD1, mSumD2, mSumD3, mSumD4;

        // from left to right
        for (size_t idx = 0; idx < sizeX / _THREAD_; idx++) {
            for (size_t th = 0; th < _THREAD_; th++) {
                index[th] = (idx * _THREAD_ + th) * sizeY;
            }
            mIndex = _mm256_loadu_si256((__m256i *) index);

            mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

            mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            
            for (size_t idy = 0; idy < sizeY; idy++) {
                // In = image
                mCurIn = _mm256_i32gather_ps(Yt, mIndex, sizeof(float));
                // PreIn = n0 * In[0] + n1 * In[1] + n2 * In[2] + n3 * In[3]
                mSumN0 = _mm256_mul_ps(mCurIn,   _mm256_set1_ps(n0));
                mSumN1 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(n1));
                mSumN2 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(n2));
                mSumN3 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(n3));
                mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
                mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                // Out = PreIn - PreOut
                mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                _mm256_storeu_ps((float *) tmpRe, mCurOut);
                for (size_t k = 0; k < _THREAD_; k++) {
                    *(tmpYt + index[k] + idy) += tmpRe[k];
                }

                // step
                mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(1));
            }
        }

        // from right to left
        for (size_t idx = 0; idx < sizeX / _THREAD_; idx++) {
            for (size_t th = 0; th < _THREAD_; th++) {
                index[th] = (idx * _THREAD_ + th + 1) * sizeY - 1;
            }
            mIndex = _mm256_loadu_si256((__m256i *) index);

            mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

            mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            
            for (size_t idy = 0; idy < sizeY; idy++) {
                // In = image
                mCurIn = _mm256_i32gather_ps(Yt, mIndex, sizeof(float));
                // PreIn = m1 * In[1] + m2 * In[2] + m3 * In[3] + m4 * In[4]
                mSumN0 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(m1));
                mSumN1 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(m2));
                mSumN2 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(m3));
                mSumN3 = _mm256_mul_ps(mPrevIn4, _mm256_set1_ps(m4));
                mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
                mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                // Out = PreIn - PreOut
                mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                _mm256_storeu_ps((float *) tmpRe, mCurOut);
                for (size_t k = 0; k < _THREAD_; k++) {
                    *(tmpYt + index[k] - idy) += tmpRe[k];
                }

                // step
                mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-1));
            }
        }

        // from top to bottom
        for (size_t idy = 0; idy < sizeY / _THREAD_; idy++) {
            for (size_t th = 0; th < _THREAD_; th++) {
                index[th] = idy * _THREAD_ + th;
            }
            mIndex = _mm256_loadu_si256((__m256i *) index);

            mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

            mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            
            for (size_t idx = 0; idx < sizeX; idx++) {
                // In = image
                mCurIn = _mm256_i32gather_ps(tmpYt, mIndex, sizeof(float));
                // PreIn = n0 * In[0] + n1 * In[1] + n2 * In[2] + n3 * In[3]
                mSumN0 = _mm256_mul_ps(mCurIn,   _mm256_set1_ps(n0));
                mSumN1 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(n1));
                mSumN2 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(n2));
                mSumN3 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(n3));
                mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
                mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                // Out = PreIn - PreOut
                mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                _mm256_storeu_ps((float *) tmpRe, mCurOut);
                for (size_t k = 0; k < _THREAD_; k++) {
                    *(Yt + index[k] + idx * sizeY) = tmpRe[k];
                }

                // step
                mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(sizeY));
            }
        }

        // from bottom to top
        for (size_t idy = 0; idy < sizeY / _THREAD_; idy++) {
            for (size_t th = 0; th < _THREAD_; th++) {
                index[th] = (sizeX - 1) * sizeY + idy * _THREAD_ + th;
            }
            mIndex = _mm256_loadu_si256((__m256i *) index);

            mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

            mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
            
            for (size_t idx = 0; idx < sizeX; idx++) {
                // In = image
                mCurIn = _mm256_i32gather_ps(tmpYt, mIndex, sizeof(float));
                // PreIn = m1 * In[1] + m2 * In[2] + m3 * In[3] + m4 * In[4]
                mSumN0 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(m1));
                mSumN1 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(m2));
                mSumN2 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(m3));
                mSumN3 = _mm256_mul_ps(mPrevIn4, _mm256_set1_ps(m4));
                mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4] 
                mSumD1 = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                mSumD2 = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                mSumD3 = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                mSumD4 = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                // Out = PreIn - PreOut
                mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                _mm256_storeu_ps((float *) tmpRe, mCurOut);
                for (size_t k = 0; k < _THREAD_; k++) {
                    *(Yt + index[k] - idx * sizeY) += tmpRe[k];
                }

                // step
                mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-sizeY));
            }
        }

        free(index);
        free(tmpRe);
        free(tmpYt);

        return;
    };
};

}
