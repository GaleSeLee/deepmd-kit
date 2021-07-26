// Copyright 2021 Graphcore Ltd.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <poplar/StackSizeDefs.hpp>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <print.h>

using namespace poplar;

DEF_STACK_USAGE(512, "_ZNSt3__16__sortIRNS_6__lessIffEEPfEEvT0_S5_T_");

static constexpr auto SPAN = poplar::VectorLayout::SPAN;


template <class T>
T dot3(T *a, T *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <class T>
void sort(int *type, T *dist, int *index, int size) {
  for (int i = 1; i < size; ++i) {
    int type_tmp = type[i];
    T   dist_tmp = dist[i];
    int index_tmp = index[i];
    for (int j = i - 1; j >= 0; --j) {
      if (type_tmp < type[j]  || 
		     (type_tmp == type[j] && 
		     (dist_tmp < dist[j]  || 
		     (dist_tmp == dist[j] && index_tmp < index[j])))) {
           type[j+1] = type[j];
           dist[j+1] = dist[j];
           index[j+1] = index[j];
           if (j == 0) {
             type[0] = type_tmp;
             dist[0] = dist_tmp;
             index[0] = index_tmp;
           }
         }
      else {
        type[j+1] = type_tmp;
        dist[j+1] = dist_tmp;
        index[j+1] = index_tmp;
        break;
      }
    }
  }
}

template <class T>
void spline5_switch (
    T & vv,
    T & dd,
    const T & xx,
    const float & rmin,
    const float & rmax)
{
  if (xx < rmin) {
    dd = 0;
    vv = 1;
  }
  else if (xx < rmax) {
    T uu = (xx - rmin) / (rmax - rmin);
    T du = 1.f / (rmax - rmin);
    vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1;
    dd = (3 * uu * uu * (-6 * uu * uu + 15 * uu - 10) + uu * uu * uu * (-12 * uu + 15)) * du;
  }
  else {
    dd = 0;
    vv = 0;
  }
}


template <class T>
class FormatVertex : public Vertex {
  public:
    Input<Vector<T>> posi;
    Input<Vector<int>> type;
    Input<Vector<int>> nei_idx_a;
    Input<Vector<int>> sec_a;

    Output<Vector<int>> info_type;
    Output<Vector<T>> info_dist;
    Output<Vector<int>> info_index;
    Output<Vector<int>> fmt_nei_idx_a;
    int i_idx;
    float rcut;

    bool compute() {
      int ii = 0;
      for (auto kk = 0; kk < nei_idx_a.size(); ++kk) {
        T diff[3];
        auto j_idx = nei_idx_a[kk];
        for (auto dd = 0; dd < 3; ++dd) {
          diff[dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
        }
        auto rr = sqrt(dot3(diff, diff));

        if (rr <= rcut) {
          info_type[kk] = type[j_idx];
          info_dist[kk] = rr;
          info_index[kk] = j_idx;
          ++ii;
        }
      }

      // sort
      int *p_type = (int*)(&info_type[0]);
      T   *p_dist = (T*)(&info_dist[0]);
      int *p_index = (int*)(&info_index[0]);
      sort(p_type, p_dist, p_index, info_type.size());

      // output
      std::memset((void*)(&fmt_nei_idx_a[0]), -1, sizeof(int) * fmt_nei_idx_a.size());
      int nei_iter[3] = {sec_a[0], sec_a[1], sec_a[2]};
      for (auto kk = 0; kk < info_type.size(); ++kk) {
        auto nei_type = info_type[kk];
        if (nei_iter[nei_type] < sec_a[nei_type + 1]) {
          fmt_nei_idx_a[nei_iter[nei_type]++] = info_index[kk];
        }
      }

      return true;
    }
};


template <class T>
class EnvMatVertex : public Vertex {
  public:
    Input<Vector<T>> posi;
    Input<Vector<int>> type;
    Input<Vector<int>> fmt_nlist_a;
    Input<Vector<int>> sec_a;

    Output<Vector<T>> descrpt_a;
    Output<Vector<T>> descrpt_a_deriv;
    Output<Vector<T>> rij_a;
    int i_idx;
    float rmin;
    float rmax;

    bool compute() {
      // compute the diff of the neighbors
      for (auto ii = 0; ii < int(sec_a.size()) - 1; ++ii) {
        for (auto jj = sec_a[ii]; jj < sec_a[ii + 1]; ++jj) {
          if (fmt_nlist_a[jj] < 0) break;
          const int & j_idx = fmt_nlist_a[jj];
          for (int dd = 0; dd < 3; ++dd) {
            rij_a[jj * 3 + dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
          }
        }
      }

      std::memset((void*)(&descrpt_a[0]), 0, sizeof(T) * descrpt_a.size());
      std::memset((void*)(&descrpt_a_deriv[0]), 0, sizeof(T) * descrpt_a_deriv.size());

      for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; ++nei_iter) {
          if (fmt_nlist_a[nei_iter] < 0) break;
          const T * rr = &rij_a[nei_iter * 3];
          T nr2 = dot3(rr, rr);
          T inr = 1.f / sqrt(nr2);
          T nr = nr2 * inr;
          T inr2 = inr * inr;
          T inr4 = inr2 * inr2;
          T inr3 = inr4 * nr;

          T sw, dsw;
          spline5_switch(sw, dsw, nr, rmin, rmax);

          int idx_deriv = nei_iter * 4 * 3;	  // 4 components time 3 directions
          int idx_value = nei_iter * 4;	      // 4 components

          // 4 value components
          descrpt_a[idx_value + 0] = 1.f / nr;
          descrpt_a[idx_value + 1] = rr[0] / nr2;
          descrpt_a[idx_value + 2] = rr[1] / nr2;
          descrpt_a[idx_value + 3] = rr[2] / nr2;

          // deriv of component 1/r
          descrpt_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[0] * inr;
          descrpt_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[1] * inr;
          descrpt_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[2] * inr;

          // deriv of component x/r2
          descrpt_a_deriv[idx_deriv + 3] = (2.f * rr[0] * rr[0] * inr4 - inr2) * sw - descrpt_a[idx_value + 1] * dsw * rr[0] * inr;
          descrpt_a_deriv[idx_deriv + 4] = (2.f * rr[0] * rr[1] * inr4	) * sw - descrpt_a[idx_value + 1] * dsw * rr[1] * inr;
          descrpt_a_deriv[idx_deriv + 5] = (2.f * rr[0] * rr[2] * inr4	) * sw - descrpt_a[idx_value + 1] * dsw * rr[2] * inr;

          // deriv of component y/r2
          descrpt_a_deriv[idx_deriv + 6] = (2.f * rr[1] * rr[0] * inr4	) * sw - descrpt_a[idx_value + 2] * dsw * rr[0] * inr;
          descrpt_a_deriv[idx_deriv + 7] = (2.f * rr[1] * rr[1] * inr4 - inr2) * sw - descrpt_a[idx_value + 2] * dsw * rr[1] * inr;
          descrpt_a_deriv[idx_deriv + 8] = (2.f * rr[1] * rr[2] * inr4	) * sw - descrpt_a[idx_value + 2] * dsw * rr[2] * inr;

          // deriv of component z/r2
          descrpt_a_deriv[idx_deriv + 9] = (2.f * rr[2] * rr[0] * inr4	) * sw - descrpt_a[idx_value + 3] * dsw * rr[0] * inr;
          descrpt_a_deriv[idx_deriv +10] = (2.f * rr[2] * rr[1] * inr4	) * sw - descrpt_a[idx_value + 3] * dsw * rr[1] * inr;
          descrpt_a_deriv[idx_deriv +11] = (2.f * rr[2] * rr[2] * inr4 - inr2) * sw - descrpt_a[idx_value + 3] * dsw * rr[2] * inr;

          // 4 value components
          descrpt_a[idx_value + 0] *= sw;
          descrpt_a[idx_value + 1] *= sw;
          descrpt_a[idx_value + 2] *= sw;
          descrpt_a[idx_value + 3] *= sw;
        }
      }

      return true;
    }
};

template class FormatVertex<float>;
template class EnvMatVertex<float>;
