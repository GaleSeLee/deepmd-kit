#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <iostream>

extern "C"
{
    int32_t custom_op_api_level = 4;
}

extern "C" poplar::program::Program ProdVirialSeA(
    poplar::Graph &graph, const std::vector<poplar::Tensor> &inputs,
    std::vector<poplar::Tensor> &outputs, const std::string &debugPrefix)
{
    using namespace poplar;
    int input_index = 0;
    const Tensor &net_deriv_tensor = inputs[input_index++];
    const Tensor &in_deriv_tensor = inputs[input_index++];
    const Tensor &rij_tensor = inputs[input_index++];
    const Tensor &nlist_tensor = inputs[input_index++];
    const Tensor &nloc_tensor = inputs[input_index++];
    const Tensor &nall_tensor = inputs[input_index++];

    int nloc = nloc_tensor.shape()[0];
    int nall = nall_tensor.shape()[0];
    int ndescrpt = net_deriv_tensor.shape()[1] / nloc;
    int nframes = net_deriv_tensor.shape()[0];
    int nnei = nlist_tensor.shape()[1] / nloc;

    Tensor virial_tensor = graph.addVariable(FLOAT, {nframes, 9}, "virial");
    Tensor atom_virial_tensor = graph.addVariable(FLOAT, {nframes, nall * 9}, "atom_virial");

    outputs.push_back(virial_tensor);
    outputs.push_back(atom_virial_tensor);

    auto cs = graph.addComputeSet(debugPrefix + "/ProdVirialVertex");
    for (int i = 0; i < nframes; i++)
    {
        auto v = graph.addVertex(cs, poputil::templateVertex("ProdVirialVertex", FLOAT));
        graph.setTileMapping(v, i);
        graph.setTileMapping(rij_tensor[i], i);
        graph.setTileMapping(net_deriv_tensor[i], i);
        graph.setTileMapping(in_deriv_tensor[i], i);
        graph.setTileMapping(nlist_tensor[i], i);
        graph.setTileMapping(virial_tensor[i], i);
        graph.setTileMapping(atom_virial_tensor[i],i);

        graph.connect(v["rij"], rij_tensor[i]);
        graph.connect(v["net_deriv"], net_deriv_tensor[i]);
        graph.connect(v["env_deriv"], in_deriv_tensor[i]);
        graph.connect(v["nlist"], nlist_tensor[i]);
        graph.connect(v["virial"],virial_tensor[i]);
        graph.connect(v["atom_virial"],atom_virial_tensor[i]);

        graph.setInitialValue(v["nall"], nall);
        graph.setInitialValue(v["nnei"], nnei);
        graph.setInitialValue(v["nloc"], nloc);
        graph.setInitialValue(v["ndescrpt"], ndescrpt);
    }
    return program::Execute(cs);

}
