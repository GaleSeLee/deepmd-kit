#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <poplar/StackSizeDefs.hpp>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <print.h>

using namespace poplar;

inline void
make_index_range(
    int &idx_start,
    int &idx_end,
    const int &nei_idx,
    const int &nnei)
{
    if (nei_idx < nnei)
    {
        idx_start = nei_idx * 4;
        idx_end = nei_idx * 4 + 4;
    }
}


template <class FPTYPE>
class ProdForceVertex_1 : public Vertex
{
public:
    Input<Vector<FPTYPE>> net_deriv;
    Input<Vector<FPTYPE>> env_deriv;

    Output<Vector<FPTYPE>> force;
    int nall;
    int nnei;
    int nloc;
    int ndescrpt;

    bool compute()
    {
        for(int aa=0;aa<force.size();aa++)
        {
            force[aa]=0;
        }

        for (int aa = 0; aa < ndescrpt; ++aa)
        {
	    force[0] -= net_deriv[aa] * env_deriv[aa * 3 + 0];
        force[1] -= net_deriv[aa] * env_deriv[aa * 3 + 1];
        force[2] -= net_deriv[aa] * env_deriv[aa * 3 + 2];
        }
        return true;
    }
};

template <class FPTYPE>
class ProdForceVertex_2 : public Vertex
{
public:
    Input<Vector<FPTYPE>> net_deriv;
    Input<Vector<FPTYPE>> env_deriv;
    Input<Vector<int>> nlist;

    Output<Vector<FPTYPE>> force;
    int nloc;
    int nall;
    int nnei;
    int ndescrpt;

    bool compute()
    {
        for (int i_idx = 0; i_idx < nloc; ++i_idx)
        {
            for (int jj = 0; jj < nnei; ++jj)
            {
                int j_idx = nlist[i_idx * nnei + jj];
                if (j_idx < 0)
                    continue;
                int aa_start, aa_end;
                make_index_range(aa_start, aa_end, jj, nnei);
                for (int aa = aa_start; aa < aa_end; ++aa)
                {
                    force[j_idx * 3 + 0] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
                    force[j_idx * 3 + 1] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
                    force[j_idx * 3 + 2] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
                }
            }
        }
	return true;
    }
};
template class ProdForceVertex_1<float>;
template class ProdForceVertex_2<float>;
