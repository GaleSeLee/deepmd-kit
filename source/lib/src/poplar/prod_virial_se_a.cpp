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

template <class T>
class ProdVirialVertex : public Vertex
{
public:
    Input<Vector<T>>net_deriv;
    Input<Vector<T>>env_deriv;
    Input<Vector<T>>rij;
    Input<Vector<int>>nlist;

    Output<Vector<T>>virial;
    Output<Vector<T>>atom_virial;
    int nloc;
    int nall;
    int nnei;
    int ndescrpt;

    bool compute()
    {
        for (int ii = 0; ii < 9; ++ii)
        {
            virial[ii] = 0.;
        }
        for (int ii = 0; ii < 9 * nall; ++ii)
        {
            atom_virial[ii] = 0.;
        }

        // compute virial of a frame
        for (int ii = 0; ii < nloc; ++ii)
        {
            int i_idx = ii;

            // deriv wrt neighbors
            for (int jj = 0; jj < nnei; ++jj)
            {
                int j_idx = nlist[i_idx * nnei + jj];
                if (j_idx < 0)
                    continue;
                int aa_start, aa_end;
                make_index_range(aa_start, aa_end, jj, nnei);
                for (int aa = aa_start; aa < aa_end; ++aa)
                {
                    T pref = -1.0 * net_deriv[i_idx * ndescrpt + aa];
                    for (int dd0 = 0; dd0 < 3; ++dd0)
                    {
                        for (int dd1 = 0; dd1 < 3; ++dd1)
                        {
                            T tmp_v = pref * rij[i_idx * nnei * 3 + jj * 3 + dd1] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd0];
                            virial[dd0 * 3 + dd1] -= tmp_v;
                            atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
                        }
                    }
                }
            }
        }
        return true;
    }
};

template class ProdVirialVertex<float>;